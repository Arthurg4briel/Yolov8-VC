# **YoloV8-vc**

Treinamento do **YOLOv8** para detecção de objetos simples utilizando técnicas de **visão computacional**, **transfer learning**, **fine-tuning** e implementação do modelo em um script local no **VS Code**.

---

## 📌 **Descrição do Projeto**

Este repositório contém todo o processo de criação de um modelo de detecção de objetos utilizando o **YOLOv8**, desde a preparação do dataset personalizado, treinamento no Google Colab, até a implementação final do modelo em um ambiente local.

O fluxo principal deste projeto envolve:

- Montagem de um **dataset próprio** com classes específicas.  
- Aplicação de **transfer learning** usando o modelo pré-treinado YOLOv8.  
- Realização de **fine-tuning** para ajustar o modelo às classes desejadas.  
- Treinamento utilizando o **Google Colab**.  
- Exportação do modelo treinado e utilização em um script Python no **VS Code** para fazer inferências em imagens.

---


## **Como baixar as dependências**

Dentro do arquivo requirements.txt, há as dependências usadas ao longo do projeto. Portanto, reduz o trabalho do usuário na instalação dessas bibliotecas.

Copie e cole este comando no terminal da sua pasta raiz do projeto:

``` bash
    pip install -r requirements.txt
```

---

## **Rodando a aplicação**

Para rodar o código, use este comando no terminal dentro da pasta do projeto.

```bash
    python app.py
 ```

## 🧠 **Tecnologias Utilizadas**

- **YOLOv8** (Ultralytics)  
- **Python 3.13.4**  
- **Google Colab**  
- **OpenCV**  
- **NumPy**  
- **VS Code**  
- **Dataset personalizado (anotações YOLO)**  

---
