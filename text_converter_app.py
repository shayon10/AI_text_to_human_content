import tkinter as tk
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextConverterApp:
    def __init__(self, master):
        self.master = master
        master.title("AI to Human Content Converter")

        self.text_entry = tk.Text(master, height=10, width=40)
        self.text_entry.pack(side=tk.LEFT, padx=10)

        self.convert_button = tk.Button(master, text="Convert", command=self.convert_text)
        self.convert_button.pack(side=tk.LEFT, padx=10)

        self.result_text = tk.Text(master, height=10, width=40, state=tk.DISABLED)
        self.result_text.pack(side=tk.LEFT, padx=10)

    def convert_text(self):
        ai_text = self.text_entry.get("1.0", tk.END)

        # Use a pre-trained GPT-2 model for text conversion
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Tokenize and generate text with lower temperature
        input_ids = tokenizer.encode(ai_text, return_tensors="pt")
        output = model.generate(input_ids, max_length=150, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

        # Decode and display the converted text
        human_text = tokenizer.decode(output[0], skip_special_tokens=True)
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, human_text)
        self.result_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = TextConverterApp(root)
    root.mainloop()
