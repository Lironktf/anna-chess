//! anna-v3 trainer: threat + pawn-pair + psq inputs, pairwise FT, multilayer output.
//! Adapted from bullet examples/advanced/main.rs (rev 629ee50). The saved layout matches
//! engine/src/nnue/v3.rs exactly (see the doc comment there). Configuration by env vars:
//!   DATA (comma-separated SF binpacks), NET_ID, SB0/SB1/SB2 (superbatches per stage), RESUME (checkpoint dir), LR1 (stage-1 start LR),
//!   THREADS (map threads), LOADER_THREADS, BUFFER_MB, SAVE_RATE, OUT_DIR, L1 (accumulator width: 1024 default, 512 half).
use bullet::{
    game::{
        inputs::{get_num_buckets, ChessBucketsMirrored, SparseInputType},
        outputs::MaterialCount,
    },
    trainer::schedule::{
        lr::{self, LrScheduler},
        wdl,
    },
    value::{loader::sfbinpack::SfBinpackLoader, save::save_to_checkpoint},
};
use bullet_trainer::{
    model::{InitSettings, ModelDefinition, ModelEvaluator, ModelInputs, ModelWeights, SavedFormat},
    optimiser::{
        adam::{AdamW, AdamWParams},
        Optimiser,
    },
    reader::ReadMapLoader,
    run::{train, DefaultDevice, TrainingSchedule, TrainingSteps},
};
use trainer::bullet_inputs::{make_inputs_mapper, three_file_band_mask, PawnPawnInputs};
use trainer::{filter, BUCKET_LAYOUT};

const L2: usize = 16;
const L3: usize = 32;
const Q0: i16 = 255;
const Q1: i16 = 128;
const INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);
const OUTPUT_BUCKETS: usize = 8;
const PP_RANGE: f32 = 127.0 / Q0 as f32; // i8 threat weights: |w| * 255 <= 127
const L1_RANGE: f32 = 127.0 / Q1 as f32; // i8 L1 weights: |w| * 128 <= 127

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let net_id: String = std::env::var("NET_ID").unwrap_or_else(|_| "anna-v3".to_string());
    let width: usize = env_or("L1", 1024);
    assert!(width % 64 == 0 && width >= 128, "width must be a multiple of 64");
    let data_paths: String = std::env::var("DATA").expect("set DATA=comma,separated,binpacks");
    let sb0: usize = env_or("SB0", 40);
    let sb1: usize = env_or("SB1", 600);
    let sb2: usize = env_or("SB2", 60);
    let map_threads: u8 = env_or("THREADS", 8);
    let loader_threads: usize = env_or("LOADER_THREADS", 8);
    let buffer_mb: usize = env_or("BUFFER_MB", 4096);
    let save_rate: usize = env_or("SAVE_RATE", 10);
    let out_dir: String = std::env::var("OUT_DIR").unwrap_or_else(|_| "checkpoints".to_string());
    let batch_size: usize = env_or("BATCH_SIZE", 16_384 * 8);
    let batches_per_sb: usize = env_or("BATCHES_PER_SB", 6104 / 8);

    let pp = PawnPawnInputs::new(three_file_band_mask());
    let psqt = ChessBucketsMirrored::new(BUCKET_LAYOUT);
    let output_buckets = MaterialCount::<OUTPUT_BUCKETS>;
    assert_eq!(pp.num_inputs(), 4560 + 59808);
    assert_eq!(psqt.num_inputs(), 768 * INPUT_BUCKETS);

    let inputs = ModelInputs::default()
        .add_sparse("stm/pp", (pp.num_inputs(), 1), pp.max_active())
        .add_sparse("ntm/pp", (pp.num_inputs(), 1), pp.max_active())
        .add_sparse("stm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("ntm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("buckets", (OUTPUT_BUCKETS, 1), 1)
        .add_dense("targets", (1, 1));

    let defn = ModelDefinition::build(&inputs, |builder, (((((stm_pp, ntm_pp), stm_psqt), ntm_psqt), output_buckets), target)| {
        let l0_pp = builder.new_affine("l0/pp/", pp.num_inputs(), width);
        let l0f = builder.new_weights("l0/fac", (width, 768), InitSettings::Zeroed);
        let psqt_init = InitSettings::Normal { mean: 0.0, stdev: (2f32 / 32.0).sqrt() };
        let mut l0_psqt = builder.new_weights("l0/psqt", (width, psqt.num_inputs()), psqt_init);
        l0_psqt = l0_psqt + l0f.repeat(psqt.num_inputs() / 768);

        let l1 = builder.new_affine("l1/", width, OUTPUT_BUCKETS * L2);
        let l2 = builder.new_affine("l2/", L2 * 2, OUTPUT_BUCKETS * L3);
        let l3 = builder.new_affine("l3/", L3, OUTPUT_BUCKETS);

        // Feature transformer half: (pp weights + psq weights) -> crelu; pairwise product of halves.
        let ft = |pp_in, psqt_in, start, end| (l0_pp.slice(start, end).forward(pp_in) + l0_psqt.slice_rows(start, end).matmul(psqt_in)).crelu();
        let stm_hidden = ft(stm_pp, stm_psqt, 0, width / 2) * ft(stm_pp, stm_psqt, width / 2, width);
        let ntm_hidden = ft(ntm_pp, ntm_psqt, 0, width / 2) * ft(ntm_pp, ntm_psqt, width / 2, width);
        let l0_out = stm_hidden.concat(ntm_hidden);
        let l0_out_norm = l0_out.reduce_sum_rows() / (width as f32);

        let l1_out = l1.forward(l0_out).select(output_buckets);
        // Dual activation: [crelu(v), crelu(v^2)] (engine: v.clamp(0,1) and (v*v).clamp(0,1)).
        let hl2 = l1_out.concat(l1_out.abs_pow(2.0)).crelu();
        let l2_out = l2.forward(hl2).select(output_buckets);
        let hl3 = l2_out.crelu();
        let l3_out = l3.forward(hl3).select(output_buckets);

        let loss = l3_out.sigmoid().squared_error(target);
        let loss = loss + 0.005 * l0_out_norm;
        (Some(loss.reduce_sum_batch()), vec![("output".to_string(), l3_out)])
    });

    let weights = ModelWeights::new(&defn, 20260912);
    let device = DefaultDevice::new(0).unwrap();
    let params = AdamWParams::default();
    let mut evaluator = ModelEvaluator::new(&defn, device.clone()).unwrap();
    let mut optimiser = Optimiser::<_, AdamW<_>>::new(defn, weights, device.clone(), params).unwrap();

    // Clipping so every quantised tensor fits its integer type.
    let l0_clip = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    optimiser.set_params_for_weight("l0/fac", l0_clip);
    optimiser.set_params_for_weight("l0/psqt", l0_clip);
    let pp_clip = AdamWParams { max_weight: PP_RANGE, min_weight: -PP_RANGE, ..Default::default() };
    optimiser.set_params_for_weight("l0/pp/w", pp_clip);
    let l1_clip = AdamWParams { max_weight: L1_RANGE, min_weight: -L1_RANGE, ..Default::default() };
    optimiser.set_params_for_weight("l1/w", l1_clip);
    // RESUME=<checkpoint dir with raw.bin + optimiser_state>: continue training from a saved run
    // (weights and Adam moments). Use SB0=0 to skip the warmup stage and LR1 for the restart LR.
    if let Ok(path) = std::env::var("RESUME") {
        optimiser.load_from_checkpoint(&format!("{path}/optimiser_state")).expect("RESUME: cannot load optimiser_state");
        println!("resumed weights and optimiser state from {path}");
    }
    let lr1_init: f32 = env_or("LR1", 1e-3);

    // Saved layout == engine/src/nnue/v3.rs: psq i16 | pp i8 | ft bias i16 | l1 w i8 (transposed) |
    // l1 b f32 | l2 w f32 (transposed) | l2 b f32 | l3 w f32 (transposed) | l3 b f32.
    let saved_format = vec![
        SavedFormat::id("l0/psqt")
            .transform(|weights, values| {
                let fac = weights.get("l0/fac").values.f32().repeat(INPUT_BUCKETS);
                assert_eq!(values.len(), fac.len());
                values.iter().zip(fac).map(|(&a, b)| a + b).collect()
            })
            .round()
            .quantise::<i16>(Q0),
        SavedFormat::id("l0/pp/w").round().quantise::<i8>(Q0),
        SavedFormat::id("l0/pp/b").round().quantise::<i16>(Q0),
        SavedFormat::id("l1/w").transpose().round().quantise::<i8>(Q1),
        SavedFormat::id("l1/b"),
        SavedFormat::id("l2/w").transpose(),
        SavedFormat::id("l2/b"),
        SavedFormat::id("l3/w").transpose(),
        SavedFormat::id("l3/b"),
    ];

    let paths: Vec<&str> = data_paths.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    println!("anna-v3 trainer | net {} | data {:?} | stages {}/{}/{} SB | batch {} x {} | width {} L2 {} L3 {}", net_id, paths, sb0, sb1, sb2, batch_size, batches_per_sb, width, L2, L3);
    let reader = SfBinpackLoader::new_concat_multiple(&paths, buffer_mb, loader_threads, filter);
    let params_tuple = (&inputs, &pp, psqt, output_buckets);

    let mut run = |stage: usize, end_superbatch: usize, lr_schedule: Box<dyn Fn(bullet_trainer::run::Step) -> f32>, mapper| {
        if end_superbatch == 0 {
            return;
        }
        train(
            &mut optimiser,
            TrainingSchedule { steps: TrainingSteps { batch_size, batches_per_superbatch: batches_per_sb, start_superbatch: 1, end_superbatch }, lr_schedule, log_rate: 128 },
            ReadMapLoader::new(reader.clone(), mapper, map_threads),
            |_, _, _| {},
            |optimiser, step| {
                let superbatch = step.superbatch();
                if superbatch.is_multiple_of(save_rate) || superbatch == step.final_superbatch() {
                    let name = format!("{net_id}-s{stage}-{superbatch}");
                    save_to_checkpoint(optimiser, &saved_format, &format!("{out_dir}/{name}"));
                    println!("Saved [{name}]");
                }
            },
        )
        .unwrap();
    };

    // Stage 0: warmup (LR ramp up then down), low WDL.
    let warm = (sb0 / 2).max(1);
    run(
        0,
        sb0,
        lr::Sequence {
            first: lr::LinearDecayLR { initial_lr: 1e-4, final_lr: 5e-3, final_superbatch: warm },
            second: lr::LinearDecayLR { initial_lr: 5e-3, final_lr: 1e-4, final_superbatch: sb0 - warm },
            first_scheduler_final_superbatch: warm,
        }
        .boxed(),
        make_inputs_mapper(params_tuple, wdl::ConstantWDL { value: 0.2 }),
    );
    // Stage 1: main run.
    run(
        1,
        sb1,
        lr::LinearDecayLR { initial_lr: lr1_init, final_lr: 1e-6, final_superbatch: sb1 }.boxed(),
        make_inputs_mapper(params_tuple, wdl::LinearWDL { start: 0.2, end: 0.5 }),
    );
    // Stage 2: short pure-WDL fine-tune at a tiny LR.
    run(
        2,
        sb2,
        lr::LinearDecayLR { initial_lr: 1e-5, final_lr: 1e-7, final_superbatch: sb2 }.boxed(),
        make_inputs_mapper(params_tuple, wdl::ConstantWDL { value: 1.0 }),
    );
    // Print the (unquantised) model's evaluation of the netcheck positions so the engine's reading of
    // quantised.bin can be compared against the trainer (layout proof).
    evaluator.load_device_weights(optimiser.weights()).unwrap();
    let evaluator_mapper = make_inputs_mapper(params_tuple, wdl::ConstantWDL { value: 0.0 });
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1",
        "rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bqkb1r/pp2bppp/2n2n2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 9",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let pos = format!("{fen} | 0 | 0.0").parse().unwrap();
        let inputs = evaluator_mapper.map(&[pos], Default::default(), 1).to_device(&device).unwrap();
        let output = evaluator.evaluate(&inputs).unwrap().get("output").unwrap();
        let [value] = output.to_host().unwrap().f32()[..] else { panic!() };
        println!("TRAINER EVAL {} : {:.1}", fen, 400.0 * value);
    }
    println!("done");
}
