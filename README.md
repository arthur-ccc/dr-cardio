# Dr. Neurônios 🫀
Sistema de Inteligência Artificial para análise automática de artigos científicos e auxílio no diagnóstico de doenças cardiovasculares.

🚨 **Aviso importante:** Este projeto tem fins acadêmicos e **não substitui a avaliação médica profissional**.


## Arquitetura do Pipeline

1. **Ingestão dos dados**  
   - Artigos advindos de bases open-source (kaglee).  
   - Download → Conversão para texto puro → Coleta de metadados.

2. **NLP (Processamento de Linguagem Natural)**  
   - Transformação do texto puro em informação estruturada.  
   - Extração de entidades biomédicas (sintomas, doenças, exames).  
   - Identificação de relações (ex.: `febre` → `cardiopatia`).  

3. **Estruturação do Conhecimento**  
   - Organização em tabelas.  
   - Consolidação de múltiplas evidências com referência aos artigos.

4. **Camada de Decisão**  
   - Implementação de **Árvore de Decisão** para diagnóstico.  
   - Justificativa: maior **explicabilidade** em comparação com sistemas especialistas.  

## Justificativas Técnicas

- **Algoritmo escolhido:** Árvore de Decisão (XgBoost)
  - Alta performance quando comparado ao tradicional CART.
  - Explicável e transparente.  
  - Adequado para representar conhecimento derivado da literatura.  
  - Fácil de validar em TDD.  

- **Alternativa descartada:** Sistema Especialista  
  - Mais trabalhoso de manter (regras manuais).  
  - Menos flexível ao crescer a base de artigos.  

- **Ferramentas previstas:**  
  - Python  
  - Testes: pytest
  - KaggleApi

## Plano de TDD

- **Testes Unitários**
  - Extração de texto e metadados dos artigos. 
  - Reconhecimento de entidades biomédicas em frases de teste.

- **Testes de Relação**
  - Sintoma ↔ Doença (ex.: "dor no peito" → "cardiopatia").

- **Testes da Árvore de Decisão**
  - Predições em casos clínicos sintéticos.
  - Explicabilidade: cada decisão deve exibir o caminho da árvore.

- **Testes de Integração**
  - Pipeline completo: artigo → entidades → árvore → diagnóstico.  


## Como Rodar (futuro)

```bash
# Clonar repositório
git clone https://github.com/seu-user/dr-neuronios

# Instalar dependências
pip install -r requirements.txt

# Rodar testes
pytest
