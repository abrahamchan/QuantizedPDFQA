# QuantizedPDFQA
A Quantized Local LLM to interactively query multiple source PDFs. No OpenAI required.

QuantizedPDFQA is powered by HuggingFace, LangChain, and Chroma.
QuantizedPDFQA uses a quantized `Flan-T5` model for the LLM, and a quantized version of `Instructor` for the sentence embedding model.
QuantizedPDFQA is designed to run entirely locally on your own workstation (2 GB total on local storage, no GPU required), so no data is sent remotely.

## Installation
QuantizedPDFQA requires Python 3+. Install all dependencies listed in requirements.txt by running this command:
```
pip install -r requirements.txt
```

## Instructions to Run
To use QuantizedPDFQA, run the following command with the path to the folder containing the PDFs.

```
python quant_pdf_qa.py [path_to_pdf_folder]
```

