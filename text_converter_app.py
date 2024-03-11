import tkinter as tk
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextConverterApp:
    def __init__(self, master):
        self.master = master
        master.title("AI to Human Content Converter")

        self.label = tk.Label(master, text="Enter AI-generated text:")
        self.label.pack()

        self.text_entry = tk.Entry(master, width=50)
        self.text_entry.pack()

        self.convert_button = tk.Button(master, text="Convert", command=self.convert_text)
        self.convert_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

    def convert_text(self):
        ai_text = self.text_entry.get()

        # Use a pre-trained GPT-2 model for text conversion
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Tokenize and generate text
        input_ids = tokenizer.encode(ai_text, return_tensors="pt")
        output = model.generate(input_ids, max_length=150, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

        # Decode and display the converted text
        human_text = tokenizer.decode(output[0], skip_special_tokens=True)
        self.result_label.config(text="Human-like Content:\n" + human_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = TextConverterApp(root)
    root.mainloop()
