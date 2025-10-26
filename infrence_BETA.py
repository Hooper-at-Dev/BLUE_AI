from model import LLAMA_3, Tokenizer
from helper_functions import weight_injector, clean_serp_data, fetch_context_from_web, generate, format_message
import torch

CONFIGURATIONS = {
  "DIM": 3072,
  "FFN_DIM": 8192,
  "N_LAYERS": 28,
  "N_HEADS": 24,
  "N_KV_HEADS": 8,
  "VOCAB_SIZE": 128256,
  "NORM_EPS": 1e-5,
  "ROPE_THETA": 500000,
  "MAX_BATCH_SIZE": 4,
  "MAX_SEQ_LEN": 6000,
  "N_KV_HEAD_REP": 24 // 8,
  "HEAD_DIM": 128
}

GREEN = "\033[92m"
RESET = "\033[0m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"


device = "cuda" if torch.cuda.is_available() else "cpu"

tok_DIR = "Weights/3B-instruct//original/tokenizer.model"
weight_DIR = "./Weights/3B-instruct/original/consolidated.00.pth"

llama = LLAMA_3(CONFIGURATIONS).to(device)
tokenizer = Tokenizer(tok_DIR)
weight_injector(llama, weight_DIR)

system_prompt = "You are a concise and factual AI assistant, and your name is Blue. You have to answer the Question asked by user."

def main():
    conversation = (
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt.strip()}\n"
        "<|eot_id|>\n"
    )

    conversation_tokens = torch.tensor(
        [tokenizer.encode(conversation, bos=True, eos=False, allowed_special="all")], 
        device=device
    )
    
    print(f"{RED}-" * 70)
    print(f"                              {BLUE}Blue AI{RESET}")
    print(f"                     {YELLOW}CREATOR: {GREEN}SARTHAK SANJEEV{RESET}")
    print(f"            {YELLOW}RUNNING ON:{RESET} {GREEN}{torch.cuda.get_device_name(0)}{RESET}")
    print(f"{RED}-{RESET}" * 70)
    print(f"{YELLOW}Commands:")
    print("  - Type your message to chat")
    print("  - Type 'exit' or 'quit' to end the conversation")
    print("  - Type 'reset' to start a new conversation")
    print("  - Type 'web on' to enable web search for all queries")
    print("  - Type 'web off' to disable web search")
    print(f"{RED}-{RESET}" * 70)
    print()
    
    web_search_enabled = False
    
    while True:
        user_input = input(f"{GREEN}You:{RESET} ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'web on':
            web_search_enabled = True
            print("\n[Web search mode enabled]\n")
            continue
            
        if user_input.lower() == 'web off':
            web_search_enabled = False
            print("\n[Web search mode disabled]\n")
            continue
            
        if user_input.lower() == 'reset':
            print("\n[Conversation reset]\n")
            conversation = (
                "<|start_header_id|>system<|end_header_id|>\n"
                f"{system_prompt.strip()}\n"
                "<|eot_id|>\n"
            )
            conversation_tokens = torch.tensor(
                [tokenizer.encode(conversation, bos=True, eos=False, allowed_special="all")], 
                device=device
            )
            llama.reset_cache()
            continue
        
        if web_search_enabled:
            print(f"\n[Searching web for context]")
            context_data = fetch_context_from_web(user_input, n_results=3)
            context_data = clean_serp_data(context_data)
            print(f"[Context retrieved: {len(context_data)} chars]\n")
            user_input = f"Use the following context to answer:\n{context_data}\n\nQuestion: {user_input}"
        
        user_msg = format_message("user", user_input)
        user_tokens = tokenizer.encode(user_msg, bos=False, eos=False, allowed_special="all")

        conversation_tokens = torch.cat([
            conversation_tokens,
            torch.tensor([user_tokens], device=device)
        ], dim=1)
        
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n"
        assistant_header_tokens = tokenizer.encode(assistant_header, bos=False, eos=False, allowed_special="all")
        conversation_tokens = torch.cat([
            conversation_tokens,
            torch.tensor([assistant_header_tokens], device=device)
        ], dim=1)
        
        
        conversation_tokens, assistant_response = generate(
            model=llama,
            tokenizer=tokenizer,
            conversation_tokens=conversation_tokens,
            max_new_tok=50,
            top_k=80,
            temp=0.9,
            device=device,
            context_len=2048
        )
        print(f"{BLUE}Blue AI:{RESET} ", end="", flush=True)
        print(assistant_response)
        print()


if __name__ == "__main__":
    main()
