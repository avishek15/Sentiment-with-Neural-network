from time import perf_counter
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from data_processing import get_data, preprocess_data, encoder
from neuralNet import NeuralNetwork, device

if __name__ == '__main__':
    context_len = 64
    n_embed = 64
    learning_rate = 1e-1
    batch_size = 32
    eval_iters = 100
    max_iters = 10000
    eval_interval = 500

    d1, d2, d3 = get_data()
    start_time = perf_counter()
    encoding_dict = preprocess_data(d1)
    print(f"It took {(perf_counter() - start_time)/60} mins to complete this step.")
    train_x = np.zeros((len(d1), context_len), dtype=np.int32)
    test_x = np.zeros((len(d2), context_len), dtype=np.int32)
    valid_x = np.zeros((len(d3), context_len), dtype=np.int32)

    # train_x = np.zeros((len(d1), context_len), dtype=np.float32)
    # test_x = np.zeros((len(d2), context_len), dtype=np.float32)
    # valid_x = np.zeros((len(d3), context_len), dtype=np.float32)

    train_y = np.zeros((len(d1), 1), dtype=np.float32)
    test_y = np.zeros((len(d2), 1), dtype=np.float32)
    valid_y = np.zeros((len(d3), 1), dtype=np.float32)

    for idx, row in tqdm(d1.iterrows()):
        vect = encoder(row['text'], encoding_dict)
        if len(vect) > context_len:
            train_x[idx] = vect[:context_len]
        else:
            train_x[idx, :len(vect)] = vect
        train_y[idx, 0] = row['label']
    for idx, row in tqdm(d2.iterrows()):
        vect = encoder(row['text'], encoding_dict)
        if len(vect) > context_len:
            test_x[idx] = vect[:context_len]
        else:
            test_x[idx, :len(vect)] = vect
        test_y[idx, 0] = row['label']
    for idx, row in tqdm(d3.iterrows()):
        vect = encoder(row['text'], encoding_dict)
        if len(vect) > context_len:
            valid_x[idx] = vect[:context_len]
        else:
            valid_x[idx, :len(vect)] = vect
        valid_y[idx, 0] = row['label']


    def get_batch(dataset: str = 'train') -> tuple[Any, Any]:
        data_x = train_x if dataset == 'train' else valid_x
        data_y = train_y if dataset == 'train' else valid_y
        ix = torch.randint(len(data_x), (batch_size,))
        x = torch.stack([torch.from_numpy(data_x[i]) for i in ix])
        y = torch.stack([torch.from_numpy(data_y[i]) for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y


    model = NeuralNetwork(context_len, n_embed).to(device)

    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        losses = torch.zeros(eval_iters)
        for dataset in ['train', 'valid']:
            for k in range(eval_iters):
                X, Y = get_batch(dataset)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[dataset] = losses.mean()
        model.train()
        return out

    last_valid = np.inf
    for it in range(max_iters):
        if it % eval_interval == 0:
            losses = estimate_loss()
            print(f"Step {it}: train loss {losses['train']:.4f}; valid loss {losses['valid']:.4f}")
            # if losses['valid'] > (last_valid * 1.1):
            #     break
            # else:
            #     last_valid = losses['valid']
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()

    tester_x, tester_y = torch.from_numpy(test_x[:batch_size]).to(device), test_y[:batch_size]
    predictions, loss = model(tester_x)
    predictions = predictions.cpu().detach().numpy()
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    print(f'Accuracy = {np.sum(predictions == tester_y)/len(tester_y)}')


