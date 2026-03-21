import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

from train import preset, set_seed
from transformer.kv_cache import KVCache
from transformer.transformer import Transformer

max_len = 512

EOS = 1

if __name__ == '__main__':
    # attn_mask = torch.ones(3, 6 + 3, dtype=torch.bool)
    # attn_mask = torch.tril(attn_mask, diagonal=6)
    # print(attn_mask)
    #
    # exit()
    global_seed = 42
    set_seed(global_seed)

    tokenizer = Tokenizer.from_file("tokenizer/trained.json")
    tokenizer.decoder = ByteLevel()

    n_layers, d_model, n_heads, _, _, _ = preset("small")
    vocab_size = tokenizer.get_vocab_size()
    model = Transformer(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        vocab_size=vocab_size,
    )
    timestamp = "0319-224725"
    model.load_state_dict(torch.load(f"weight/gpt-{timestamp}.pt"))

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    model.to(device)
    model.eval()

    kv_cache = KVCache(
        n_layers=n_layers,
        batch_size=1,
        n_heads=n_heads,
        max_len=max_len,
        d_head=d_model // n_heads,
        device=device,
        dtype=dtype
    )

    temperature = 0.8

    with torch.no_grad():
        while True:
            set_seed(global_seed)
            prompt_str = input("Prompt: ")
            encoded = tokenizer.encode(prompt_str, add_special_tokens=False)

            inputs = torch.tensor(encoded.ids, device=device, dtype=torch.long).view(1, -1)
            text_len = inputs.shape[1]

            print(prompt_str, end="", flush=True)

            while text_len < max_len:
                with torch.autocast("cuda", dtype):
                    logits = model(inputs, kv_cache=kv_cache)[:, -1, :]

                probs = torch.softmax(logits / temperature, dim=-1)
                outputs = torch.multinomial(probs, num_samples=1)
                if outputs[0, 0].item() == EOS:
                    print("\n=== Output Finished ===\n", flush=True)
                    break
                decoded = tokenizer.decode(outputs[0].tolist())
                print(decoded, end="", flush=True)

                text_len += 1
                inputs = outputs.view(1, -1)

            if text_len == max_len:
                print("\n=== Output Interrupted ===\n", flush=True)

            kv_cache.clear()
