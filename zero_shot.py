import os
import random
import torch
from datasets import load_from_disk
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import SystemMessage, HumanMessage

file = "./results/zero_shot_llama.txt"

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Ollama
llm = OllamaLLM(base_url='http://crai-04.dei.uc.pt:8080/', model='llama3.3:latest', temperature=0.7, num_predict=256)

# Load dataset
dataset_dir = './processed_datasets'
dataset = load_from_disk(os.path.join(dataset_dir, "test"))

# Check if GPU is available
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected. Running on CPU.")

def zero_shot(text):
    # System message
    system_prompt = (
        "Um trocadilho é uma brincadeira linguística baseada em ambiguidade sonora ou de significado. "
        "Envolve o uso de recursos como homonímia, paronímia, polissemia e outros jogos de palavras, "
        "explorando semelhanças fonéticas ou semânticas para criar efeitos de humor, ambiguidade ou surpresa. "
        "O objetivo é gerar múltiplos sentidos ou associações inusitadas a partir de expressões comuns.\n\n"

        "Você é um comediante português com um talento excepcional para criar piadas inteligentes e trocadilhos criativos. "
        "Sua missão é transformar um texto comum em uma versão bem-humorada, incorporando trocadilhos ou jogos de palavras sempre que possível. "
        "Mantenha o conteúdo original o mais intacto possível, alterando apenas o necessário para gerar humor.\n\n"

        "REGRAS:\n"
        "- Não explique o trocadilho nem comente o resultado.\n"
        "- Responda apenas com o texto modificado, já com os trocadilhos incorporados.\n"
        "- Seja criativo, mas preserve a estrutura e o contexto do texto original."
        "- Adicione sempre um trocadilho, focando na criação de humor."
    )


    # Construct the user prompt
    user_prompt = (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Transforma o seguinte texto numa piada:\n\"{text}\"\n"
        "<|eot_id|>"
    )

    # Create formatted messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # Generate response
    response = llm.invoke(messages)
    
    if "<think>" in response:
        filtered_response = response.split("<think>")[0]
        filtered_response += response.split("</think>")[1]

        return filtered_response
    
    return response

# Generate predictions
predictions = []

for _ in range(50):
    index = random.randint(0, len(dataset) - 1)
    entry = dataset[index]
    text = entry["original"]
    generated_text = zero_shot(text)

    predictions.append({
        "original_text": text,
        "generated_text": generated_text
    })

print(f"Results saved to {file}")
 
# Save results
with open(file, "w" , encoding="utf-8") as f:
    for i in predictions:
        f.write(f"Original: {i['original_text']}\n")
        f.write(f"Transformed: {i['generated_text']}\n")
        f.write("="*80 + "\n")
