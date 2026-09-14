#!/usr/bin/env bash
# Overnight chain (2026-09-13): two SPRTs of the policy net on the lazy binary, one after the other.
cd /home/liron/Desktop/chess
timeout 5h scripts/sprt.sh sprt/bin/anna_pol3 sprt/bin/anna_pol3 -N "option.PolicyFile=/home/liron/Desktop/chess/runs/policy/policy-v1.bin option.PolicyScale=2000" -B "option.PolicyScale=0" -t 8+0.08 -c 4 -e 0 5 -n policy2000b > sprt/policy2000b.out 2>&1
timeout 5h scripts/sprt.sh sprt/bin/anna_pol3 sprt/bin/anna_pol3 -N "option.PolicyFile=/home/liron/Desktop/chess/runs/policy/policy-v1.bin option.PolicyLmr=150" -B "option.PolicyLmr=0" -t 8+0.08 -c 4 -e 0 5 -n policylmr150 > sprt/policylmr150.out 2>&1
touch sprt/policy_chain.done
