A) Install
    pip install torch torchvision numpy pandas scikit-learn

B) Train (reproduce your pipeline)

    Data layout:

        processed_ts/
          train/*.csv
          valid/*.csv
          test/*.csv
        ../marinelabs-ml-deepsense-wake-detection-8ea2af3a69ac/data/vessel_NO_wake_timeseries/
          train/*.txt
          valid/*.txt
          test/*.txt


Run training:

    python lstm_train_filelevel.py \
      --pos_root processed_ts \
      --neg_root ../marinelabs-ml-deepsense-wake-detection-8ea2af3a69ac/data/vessel_NO_wake_timeseries \
      --out runs-lstm-file \
      --epochs 30 --batch 64


    This will:
        resample all files to 3000 samples,
        fit a StandardScaler on the train set,
        train an LSTM (2 layers, bidirectional, hidden=128),
        sweep the decision threshold on the validation set for best F1,
        save runs-lstm-file/lstm_file_best.pt containing model_state + scaler stats + best threshold.

C) Inference on new samples

    Put new files (CSV or TXT) in a folder, e.g. Dataset/lstm_filelevel_samples/.
    pip install torch torchvision numpy pandas scikit-learn matplotlib

    python lstm_infer_filelevel.py \
      --weights lstm_file_best_model.pt \
      --inputs ../Dataset/lstm_filelevel_samples \
      --out predictions_lstm_file \
      --copy_positives


Outputs:

    predictions_lstm_file/predictions.json
    predictions_lstm_file/predictions.csv
    predictions_lstm_file/positives/ (only if --copy_positives) # for every input file classified as wake (positive), the script will also copy the raw input file into a subfolder under your output directory, for who just want to quickly see "all the files predicted as containing wakes
