import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tkinter import messagebox, ttk
from tkinter import *
from tkinter import simpledialog
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
from string import punctuation
from os import listdir 
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy.linalg import norm
from numpy import dot
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import concatenate
import nltk
nltk.download('stopwords')
from numpy import array
import matplotlib.pyplot as plt
from datetime import datetime

class ModernButton(tk.Button):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.config(
            bg='#23232A',  # Dark background
            fg='#F8F8F2',  # Light text
            font=('Helvetica', 10, 'bold'),
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor='hand2',
            activebackground='#44475A',  # Slightly lighter on hover
            activeforeground='#FFB86C'   # Accent color on hover
        )
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def on_enter(self, e):
        self['bg'] = '#44475A'
        self['fg'] = '#FFB86C'

    def on_leave(self, e):
        self['bg'] = '#23232A'
        self['fg'] = '#F8F8F2'


def dark_style(widget):
    # Recursively apply dark theme to all widgets
    if isinstance(widget, (tk.Frame, tk.LabelFrame, tk.Toplevel, tk.Tk)):
        widget.configure(bg='#181A20')
    elif isinstance(widget, tk.Label):
        widget.configure(bg='#181A20', fg='#F8F8F2', font=('Helvetica', 11, 'bold'))
    elif isinstance(widget, tk.Entry):
        widget.configure(bg='#282A36', fg='#F8F8F2', insertbackground='#F8F8F2', relief=tk.FLAT, font=('Helvetica', 10))
    elif isinstance(widget, tk.Text):
        widget.configure(bg='#282A36', fg='#F8F8F2', insertbackground='#F8F8F2', relief=tk.FLAT, font=('Helvetica', 10))
    elif isinstance(widget, tk.Button):
        pass  # Already styled by ModernButton
    for child in widget.winfo_children():
        dark_style(child)


def create_main_window():
    root = tk.Tk()
    root.title("Convolutional Neural Network Based Text Steganalysis")
    root.geometry("1100x650")
    root.configure(bg='#F8F8F2')
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TLabel', background='#F8F8F2', foreground='#23242B', font=('Helvetica', 11, 'bold'))
    style.configure('TFrame', background='#F8F8F2')
    style.configure('TButton', background='#FFFFFF', foreground='#23242B', font=('Helvetica', 10, 'bold'))
    style.map('TButton', background=[('active', '#E0E0E0')], foreground=[('active', '#7E57C2')])

    # --- Main container ---
    container = tk.Frame(root, bg='#F8F8F2')
    container.pack(fill='both', expand=True, padx=40, pady=40)

    # --- Title ---
    title_label = tk.Label(container, text="Convolutional Neural Network Based\nText Steganalysis", font=('Helvetica', 26, 'bold'), bg='#F8F8F2', fg='#23242B')
    title_label.pack(pady=(0, 30))

    # --- Two columns ---
    columns = tk.Frame(container, bg='#F8F8F2')
    columns.pack(fill='both', expand=True)

    # --- Left: Actions ---
    left = tk.Frame(columns, bg='#FFFFFF', bd=0, relief=tk.FLAT, highlightthickness=0)
    left.pack(side=tk.LEFT, fill='y', expand=False, padx=(0, 20), pady=0)

    # File upload area (visual only)
    upload_frame = tk.Frame(left, bg='#FFFFFF', bd=1, relief=tk.RIDGE)
    upload_frame.pack(pady=(0, 25), padx=10, fill='x')
    upload_icon = tk.Label(upload_frame, text="", font=('Helvetica', 32), bg='#FFFFFF', fg='#7E57C2')
    upload_icon.pack(pady=(10, 0))
    upload_label = tk.Label(upload_frame, text="Drag & drop a file here, or click to select a file", font=('Helvetica', 10), bg='#FFFFFF', fg='#23242B')
    upload_label.pack(pady=(0, 10), padx=10)
    upload_frame.bind('<Button-1>', lambda e: upload_dataset())
    upload_icon.bind('<Button-1>', lambda e: upload_dataset())
    upload_label.bind('<Button-1>', lambda e: upload_dataset())

    # Buttons (original terms)
    ModernButton(left, text="Upload Dataset", command=upload_dataset).pack(pady=8, padx=20, fill=tk.X)
    ModernButton(left, text="Preprocess Data", command=preprocess).pack(pady=8, padx=20, fill=tk.X)
    ModernButton(left, text="Train CNN", command=cnn).pack(pady=8, padx=20, fill=tk.X)
    ModernButton(left, text="Analyze Text", command=predict).pack(pady=8, padx=20, fill=tk.X)
    ModernButton(left, text="Show Accuracy Graph", command=lambda: show_accuracy_bar(accuracy if 'accuracy' in globals() else 0)).pack(pady=8, padx=20, fill=tk.X)

    # --- Right: Text area only ---
    right = tk.Frame(columns, bg='#F8F8F2', bd=0, relief=tk.FLAT, highlightthickness=0)
    right.pack(side=tk.LEFT, fill='both', expand=True, padx=(20, 0), pady=0)

    # Text area for results
    text_frame = tk.Frame(right, bg='#F8F8F2', relief=tk.RAISED, borderwidth=1)
    text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Add title to text frame
    tk.Label(
        text_frame,
        text="Output",
        font=('Helvetica', 13, 'bold'),
        bg='#F8F8F2', fg='#23242B'
    ).pack(anchor='w', padx=10)

    # Text widget with scrollbar
    global text
    text = tk.Text(
        text_frame,
        font=('Consolas', 11),
        wrap=tk.WORD,
        bg='#FFFFFF', fg='#23242B',
        borderwidth=0, highlightthickness=0
    )
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    scrollbar = tk.Scrollbar(text_frame, command=text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text.configure(yscrollcommand=scrollbar.set)

    # --- Status bar ---
    status_frame = tk.Frame(root, bg='#F8F8F2', height=30)
    status_frame.pack(fill=tk.X, side=tk.BOTTOM)
    global status_label
    status_label = tk.Label(
        status_frame,
        text="System Ready",
        bg='#F8F8F2',
        fg='#23242B',
        font=('Helvetica', 10)
    )
    status_label.pack(side=tk.LEFT, padx=10)

    return root

def show_status(message, is_error=False):
    status_label.config(
        text=message,
        fg='#D32F2F' if is_error else '#23242B'  # Red for errors, dark purple for success
    )

def upload_dataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    if filename:
        text.delete('1.0', END)
        text.insert(END, "╔═══════════════════════════════════════════════════\n")
        text.insert(END, "║ Dataset Upload Status\n")
        text.insert(END, "╠═══════════════════════════════════════════════════\n")
        text.insert(END, f"║ Location: {filename}\n")
        text.insert(END, "║ Status: Successfully loaded\n")
        text.insert(END, "╚═══════════════════════════════════════════════════\n")
        show_status("Dataset loaded successfully!")

def preprocess():
    global trainX, trainy
    if not filename:
        messagebox.showerror("Error", "Please upload dataset first!")
        return
    try:
        docs1 = process_docs(filename+'/topic1', True)
        docs2 = process_docs(filename+'/topic2', True)
        trainX = docs1 + docs2
        trainy = [0 for _ in range(len(docs1))] + [1 for _ in range(len(docs2))]
        
        text.delete('1.0', END)
        text.insert(END, "╔═══════════════════════════════════════════════════\n")
        text.insert(END, "║ Preprocessing Results\n")
        text.insert(END, "╠═══════════════════════════════════════════════════\n")
        text.insert(END, f"║ Normal texts processed: {len(docs1)}\n")
        text.insert(END, f"║ Stego texts processed: {len(docs2)}\n")
        text.insert(END, f"║ Total samples: {len(trainX)}\n")
        text.insert(END, "╚═══════════════════════════════════════════════════\n")
        
        show_status("Preprocessing completed successfully!")
    except Exception as e:
        show_status(f"Error in preprocessing: {str(e)}", True)

def cnn():
    global accuracy, trainX, model, tokenizer, length
    if not trainX:
        messagebox.showerror("Error", "Please preprocess data first!")
        return
    try:
        text.delete('1.0', END)
        tokenizer = create_tokenizer(trainX)
        length = max_length(trainX)
        vocab_size = len(tokenizer.word_index) + 1
        
        text.insert(END, "╔═══════════════════════════════════════════════════\n")
        text.insert(END, "║ Model Configuration\n")
        text.insert(END, "╠═══════════════════════════════════════════════════\n")
        text.insert(END, f"║ Max document length: {length}\n")
        text.insert(END, f"║ Vocabulary size: {vocab_size}\n")
        text.insert(END, "╚═══════════════════════════════════════════════════\n\n")
        
        trainX_processed = encode_text(tokenizer, trainX, length)
        
        # Build and train model
        inputs1 = Input(shape=(length,))
        embedding1 = Embedding(vocab_size, 100)(inputs1)
        conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)
        
        inputs2 = Input(shape=(length,))
        embedding2 = Embedding(vocab_size, 100)(inputs2)
        conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
        
        inputs3 = Input(shape=(length,))
        embedding3 = Embedding(vocab_size, 100)(inputs3)
        conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        
        merged = concatenate([flat1, flat2, flat3])
        dense1 = Dense(10, activation='relu')(merged)
        outputs = Dense(1, activation='sigmoid')(dense1)
        
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Training progress bar
        progress_var = DoubleVar() 
        progress_bar = ttk.Progressbar(
            root,
            style="Custom.Horizontal.TProgressbar",
            variable=progress_var,
            maximum=10
        )
        progress_bar.pack(fill=X, padx=20, pady=10)
        
        text.insert(END, "╔═══════════════════════════════════════════════════\n")
        text.insert(END, "║ Training Progress\n")
        text.insert(END, "╠═══════════════════════════════════════════════════\n")
        
        class ProgressCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_var.set(epoch + 1)
                root.update()
                text.insert(END, f"║ Epoch {epoch+1}/10 - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}\n")
                text.see(END)

        model.fit([trainX_processed,trainX_processed,trainX_processed], array(trainy), epochs=10, batch_size=1, 
                 callbacks=[ProgressCallback()])
        
        loss, acc = model.evaluate([trainX_processed,trainX_processed,trainX_processed], array(trainy), verbose=0)
        accuracy = acc * 100
        
        text.insert(END, "╠═══════════════════════════════════════════════════\n")
        text.insert(END, f"║ Final Results:\n")
        text.insert(END, f"║ Model Accuracy: {accuracy:.2f}%\n")
        text.insert(END, "╚═══════════════════════════════════════════════════\n")
        
        show_status(f"Model trained successfully! Accuracy: {accuracy:.2f}%")
        
        progress_bar.destroy()
        
        show_accuracy_bar(accuracy)
        
    except Exception as e:
        show_status(f"Error in training: {str(e)}", True)
        print(f"Detailed error: {str(e)}")

def predict():
    global model, tokenizer, length
    try:
        input_text = simpledialog.askstring(
            "Text Analysis",
            "Enter text for steganalysis detection:",
            parent=root
        )
        if input_text:
            testX = process_text(input_text)
            testX_encoded = encode_text(tokenizer, testX, length)
            _ = model.predict([testX_encoded,  testX_encoded, testX_encoded])  # prediction not shown
            text.delete('1.0', tk.END)
            steg_message_cap = extract_steg_message(input_text)
            steg_message_last = extract_last_letter_steg(input_text)
            steg_message_spaces = extract_spaces_binary_steg(input_text)
            text.insert(END, f"Steg message (Capitals): {steg_message_cap}\n")
            text.insert(END, f"Steg message (Last Letters): {steg_message_last}\n")
            text.insert(END, f"Steg message (Spaces/Binary): {steg_message_spaces}\n")
            show_status("Steganographic messages extracted.")
    except Exception as e:
        show_status(f"Error in analysis: {str(e)}", True)

def extract_steg_message(text):
    import re
    # Get the first sentence (up to first period, exclamation, or question mark)
    sentence = re.split(r'[.!?]', text)[0]
    # Extract all words whose first letter is uppercase (anywhere in the sentence)
    words_list = [w for w in sentence.split() if w and w[0].isalpha() and w[0][0].isupper()]
    hidden = ''.join([w[0] for w in words_list])
    return hidden

def extract_last_letter_steg(text):
    import re
    sentence = re.split(r'[.!?]', text)[0]
    words = [w for w in sentence.split() if w and w[-1].isalpha()]
    hidden = ''.join([w[-1] for w in words])
    return hidden

def extract_spaces_binary_steg(text):
    # Hidden in extra spaces: double space = 1, single space = 0, decode as binary
    import re
    # Only consider the first sentence
    sentence = re.split(r'[.!?]', text)[0]
    # Find all space runs
    space_runs = re.findall(r'( +)', sentence)
    binary = ''
    for run in space_runs:
        if len(run) == 2:
            binary += '1'
        elif len(run) == 1:
            binary += '0'
    # Convert binary string to ASCII text
    chars = []
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return ''.join(chars)

def show_accuracy_bar(accuracy):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 5))
    plt.bar(['Accuracy'], [accuracy/100], color='#FFB86C')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.grid(axis='y', color='#44475A', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Keep existing helper functions
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens)
    return tokens

def process_docs(directory, is_train):
    documents = list()
    for filename in listdir(directory):
        path = directory + '/' + filename
        doc = load_doc(path)
        tokens = clean_doc(doc)
        documents.append(tokens)
    return documents

def process_text(text):
    documents = list()
    tokens = clean_doc(text)
    documents.append(tokens)
    return documents
    
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max([len(s.split()) for s in lines])

def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded    

if __name__ == "__main__":
    global root
    root = create_main_window()
    root.mainloop()
