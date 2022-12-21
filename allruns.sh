for SEED in 1 2 3; do
  for TASK in paraphrase translation captioning; do
    if [[ ! -f models/${TASK}/downsample-n-none/${SEED}/${TASK}_none_e25.pt ]]; then
      mkdir -p models/${TASK}/downsample-n-none/${SEED};
      python3 run.py ${TASK} \
        --models-dir models/${TASK}/downsample-n-none/${SEED}/ \
        --logs-dir logs/${TASK}/downsample-n-none/${SEED} \
        --train-path data/${TASK}/${SEED}/train \
        --downsample;
    fi;
    for NOISE in 0.5 1.0 1.5; do
      if [[ ! -f models/${TASK}/downsample-n-${NOISE}/${SEED}/${TASK}_none_e25.pt ]]; then
        mkdir -p models/${TASK}/downsample-n-${NOISE}/${SEED};
        python3 run.py ${TASK} \
          --models-dir models/${TASK}/downsample-n-${NOISE}/${SEED}/ \
          --logs-dir logs/${TASK}/downsample-n-${NOISE}/${SEED} \
          --train-path data/${TASK}/${SEED}/train \
          --noise ${NOISE} \
          --downsample;
      fi;
    done;
  done;
    for GROUNDING_TASK in translation captioning; do
      for MODE in multitask multimodal; do
        if [[ ! -f models/paraphrase-${GROUNDING_TASK}/${MODE}-downsample-n-none/${SEED}/${TASK}_${MODE}_e25.pt ]]; then
          mkdir -p models/paraphrase-${GROUNDING_TASK}/${MODE}-downsample-n-none/${SEED}/;
          python3 run.py paraphrase ${GROUNDING_TASK} \
            --models-dir models/paraphrase-${GROUNDING_TASK}/${MODE}-downsample-n-none/${SEED}/ \
            --logs-dir logs/paraphrase-${GROUNDING_TASK}/${MODE}-downsample-n-none/${SEED}/ \
            --train-path data/paraphrase-${GROUNDING_TASK}/${SEED}/train \
            --combination-mode ${MODE} \
            --downsample;
        fi;
        for NOISE in 0.5 1.0 1.5; do
          if [[ ! -f models/paraphrase-${GROUNDING_TASK}/${MODE}-downsample-n-${NOISE}/${SEED}/${TASK}_${MODE}_e25.pt ]]; then
            mkdir -p models/paraphrase-${GROUNDING_TASK}/${MODE}-downsample-n-${NOISE}/${SEED}/;
            python3 run.py paraphrase ${GROUNDING_TASK} \
              --models-dir models/paraphrase-${GROUNDING_TASK}/${MODE}-downsample-n-${NOISE}/${SEED}/ \
              --logs-dir logs/paraphrase-${GROUNDING_TASK}/${MODE}-downsample-n-${NOISE}/${SEED}/ \
              --train-path data/paraphrase-${GROUNDING_TASK}/${SEED}/train \
              --combination-mode ${MODE} \
              --noise ${NOISE} \
              --downsample;
          fi;
        done;
      done;
    done;
  done;
