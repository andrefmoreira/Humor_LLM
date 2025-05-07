import os
import random
import torch
from datasets import load_from_disk
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import SystemMessage, HumanMessage


file = "./results/few_shot_lamma.txt"

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Ollama
llm = OllamaLLM(
    base_url="http://crai-04.dei.uc.pt:8080/",
    model="llama3.3:latest",
    temperature=0.7,
    num_predict=512,
)

# Load dataset
dataset_dir = "./processed_datasets"
dataset = load_from_disk(os.path.join(dataset_dir, "test"))
n_examples = 10

# Check if GPU is available
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected. Running on CPU.")


def create_examples(dataset, n_examples):
    """Selects few-shot examples and removes them from the dataset."""
    indices = random.sample(range(len(dataset)), n_examples)
    examples = {"original": [], "transformed": []}

    for i in indices:
        entry = dataset[i]
        examples["original"].append(entry["original"])
        examples["transformed"].append(entry["transformed"])

    # Remove selected examples by filtering
    dataset = dataset.select([i for i in range(len(dataset)) if i not in indices])
    return examples, dataset


def few_shot(text, examples):
    """Generates a humorous transformation using few-shot learning."""

    # Format examples for the prompt
    example_string = ""
    for i in range(len(examples["original"])):
        example_string += f"Original: {examples['original'][i]}\n"
        example_string += f"Transformed: {examples['transformed'][i]}\n"
        example_string += "=" * 80 + "\n"

    # System message
    system_prompt = (
        "Um trocadilho é uma brincadeira linguística baseada em ambiguidade sonora ou de significado. "
        "Envolve o uso de recursos como homonímia, paronímia, polissemia e outros jogos de palavras, "
        "explorando semelhanças fonéticas ou semânticas para criar efeitos de humor, ambiguidade ou surpresa. "
        "O objetivo é gerar múltiplos sentidos ou associações inusitadas a partir de expressões comuns.\n\n"
        "Você é um comediante português com um talento excepcional para criar piadas inteligentes e trocadilhos criativos. "
        "Sua missão é transformar um texto comum em uma versão com humor, incorporando trocadilhos ou jogos de palavras sempre que possível. "
        "Os trocadilhos devem ser criativos e interessantes, e a piada deve conseguir fazer o leitor rir. "
        "Mantenha o conteúdo original o mais intacto possível, alterando apenas o necessário para gerar humor.\n\n"
        "REGRAS:\n"
        "- Não explique o trocadilho nem comente o resultado.\n"
        "- Responda apenas com o texto modificado, já com os trocadilhos incorporados.\n"
        "- Seja criativo, mas preserve a estrutura e o contexto do texto original.\n"
        "- Adicione sempre um trocadilho substituindo e alterando palavras existentes."
    )

    # User message with examples and input text
    user_prompt = (
        "Exemplos:\n"
        f"{example_string}\n\n"
        f'Transforma o seguinte texto numa piada:\n"{text}"'
    )

    # Format messages for Ollama
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    # Generate response
    response = llm.invoke(messages)

    if "<think>" in response:
        filtered_response = response.split("<think>")[0]
        filtered_response += response.split("</think>")[1]

        return filtered_response

    return response


# Generate predictions
examples, dataset = create_examples(dataset, n_examples)
num_predictions = 150
entries_to_use = dataset.select(range(num_predictions))
predictions = []

for entry in entries_to_use:
    text = entry["original"]
    generated_text = few_shot(text, examples)
    predictions.append({"original_text": text, "generated_text": generated_text})

# Remover os usados da base de dados
dataset = dataset.select(range(num_predictions, len(dataset)))

print(f"Results saved to {file}")

with open(file, "w+", encoding="utf-8") as f:
    for i in predictions:
        f.write(f"Original: {i['original_text']}\n")
        f.write(f"Transformed: {i['generated_text']}\n")
        f.write("=" * 80 + "\n")
