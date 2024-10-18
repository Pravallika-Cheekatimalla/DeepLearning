import sys
import torch
import json
import pickle
from model_seq2seq import TestDataset, test, MODELS, encoderRNN, decoderRNN, attention
from torch.utils.data import DataLoader
from bleu_eval import BLEU

# Load the model
def load_model(model_path):
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    return model.cuda()  # Move model to GPU if available

# Write predictions to output file
def save_predictions(predictions, output_path):
    with open(output_path, 'w') as f:
        for id, caption in predictions:
            f.write(f'{id},{caption}\n')

# Load test labels and compute BLEU score
def compute_bleu_score(test_labels_path, predictions_path):
    with open(test_labels_path) as f:
        test_labels = json.load(f)

    results = {}
    with open(predictions_path) as f:
        for line in f:
            line = line.rstrip()
            test_id, caption = line.split(',', 1)
            results[test_id] = caption

    bleu_scores = []
    for item in test_labels:
        score_per_video = []
        ground_truth_captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(BLEU(results[item['id']], ground_truth_captions, True))
        bleu_scores.append(score_per_video[0])

    return sum(bleu_scores) / len(bleu_scores)

# Main function
if __name__ == "__main__":
    input_data_path = sys.argv[1]
    output_file_path = sys.argv[2]
    
    dataset = TestDataset(input_data_path)
    data_loader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=8)

    # Load index-to-word mapping
    with open('i2w.pickle', 'rb') as handle:
        index_to_word = pickle.load(handle)

    # Load model
    seq2seq_model = load_model('SavedModel/model0.h5')

    # Evaluate model and get predictions
    predictions = test(data_loader, seq2seq_model, index_to_word)

    # Save predictions
    save_predictions(predictions, output_file_path)

    # Compute BLEU score
    average_bleu = compute_bleu_score('/home/pcheeka/HW2/MLDS_hw2_1_data/testing_label.json', output_file_path)
    print(f"Average BLEU score is {average_bleu:.4f}")
