//! NNUE trainer for the engine: (768 x 16 king buckets, mirrored -> 1024)x2 -> 1 x 8 output buckets.
//! Must stay in sync with engine/src/nnue/mod.rs (L1, BUCKET_LAYOUT, OUTPUT_BUCKETS, QA, QB, SCALE).
use bullet::{
    game::{
        inputs::{get_num_buckets, ChessBucketsMirrored},
        outputs::MaterialCount,
    },
    nn::{
        optimiser::{AdamW, AdamWParams},
        InitSettings, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};

use trainer::{filter, BUCKET_LAYOUT, L1, OUTPUT_BUCKETS, QA, QB, SCALE};
const INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    // Configuration via env vars so the same binary serves the local smoke test and the paid run.
    let net_id: String = std::env::var("NET_ID").unwrap_or_else(|_| "fable-v1".to_string());
    let data_paths: String = std::env::var("DATA").expect("set DATA=comma,separated,binpack,paths (Stockfish binpack format)");
    let superbatches: usize = env_or("SUPERBATCHES", 400);
    let batches_per_sb: usize = env_or("BATCHES_PER_SB", 6104);
    let batch_size: usize = env_or("BATCH_SIZE", 16384);
    let lr0: f32 = env_or("LR", 0.001);
    let wdl_start: f32 = env_or("WDL_START", 0.3);
    let wdl_end: f32 = env_or("WDL_END", 0.5);
    let threads: usize = env_or("THREADS", 4);
    let loader_threads: usize = env_or("LOADER_THREADS", 4);
    let buffer_mb: usize = env_or("BUFFER_MB", 1024);
    let save_rate: usize = env_or("SAVE_RATE", 10);
    let out_dir: String = std::env::var("OUT_DIR").unwrap_or_else(|_| "checkpoints".to_string());
    let resume: Option<String> = std::env::var("RESUME").ok();

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&[
            // Feature transformer: merge the factoriser, column-major -> one contiguous column per feature.
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            // Output layer transposed to [bucket][2*L1] for inference.
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(L1, 768), InitSettings::Zeroed);
            let mut l0 = builder.new_affine("l0", 768 * INPUT_BUCKETS, L1);
            l0.weights = l0.weights + l0f.repeat(INPUT_BUCKETS);
            let l1 = builder.new_affine("l1", 2 * L1, OUTPUT_BUCKETS);
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden).select(output_buckets)
        });

    // Weight clipping so that the i16 quantisation cannot overflow (|w| * QA < 32767 and |w| * QB < 127*...).
    let stricter = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter);
    trainer.optimiser.set_params_for_weight("l0f", stricter);

    let schedule = TrainingSchedule {
        net_id: net_id.clone(),
        eval_scale: SCALE,
        steps: TrainingSteps { batch_size, batches_per_superbatch: batches_per_sb, start_superbatch: 1, end_superbatch: superbatches },
        wdl_scheduler: wdl::LinearWDL { start: wdl_start, end: wdl_end },
        lr_scheduler: lr::CosineDecayLR { initial_lr: lr0, final_lr: lr0 * 0.3f32.powi(5), final_superbatch: superbatches },
        save_rate,
    };

    let settings = LocalSettings { threads, test_set: None, output_directory: &out_dir, batch_queue_size: 64 };

    if let Some(path) = resume {
        println!("resuming from checkpoint {}", path);
        trainer.load_from_checkpoint(&path);
    }

    // Stockfish-binpack loader with the standard filter used for Leela T80 data.
    use loader::sfbinpack::SfBinpackLoader;
    let paths: Vec<&str> = data_paths.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    println!("net {} | data {:?} | superbatches {} | arch (768x{}hm->{})x2->1x{} | QA {} QB {} scale {}",
        net_id, paths, superbatches, INPUT_BUCKETS, L1, OUTPUT_BUCKETS, QA, QB, SCALE);
    let data_loader = SfBinpackLoader::new_concat_multiple(&paths, buffer_mb, loader_threads, filter);

    trainer.run(&schedule, &settings, &data_loader);
}
