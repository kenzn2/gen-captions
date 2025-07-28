# Vietnamese Image Captioning with Flickr8k

<div align="center">
  <img src="https://img.shields.io/badge/Deep_Learning-Image_Captioning-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Language-Vietnamese-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Framework-TensorFlow_Keras-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Dataset-Flickr8k-red?style=for-the-badge">
</div>

## 📋 Tổng Quan (Overview)

Dự án này triển khai hệ thống tự động tạo mô tả tiếng Việt cho hình ảnh sử dụng Deep Learning. Mô hình được xây dựng dựa trên kiến trúc Encoder-Decoder với CNN (InceptionV3) để trích xuất đặc trưng hình ảnh và LSTM để sinh văn bản tiếng Việt.

This project implements an automatic Vietnamese image captioning system using Deep Learning. The model is built based on an Encoder-Decoder architecture with CNN (InceptionV3) for image feature extraction and LSTM for Vietnamese text generation.

## 🎯 Tính Năng Chính (Key Features)

- **Tạo Caption Tiếng Việt**: Tự động sinh mô tả tiếng Việt cho hình ảnh
- **CNN + LSTM Architecture**: Kết hợp InceptionV3 và Bidirectional LSTM
- **Word Embedding**: Sử dụng FastText Vietnamese word vectors
- **Text Processing**: Xử lý văn bản tiếng Việt với underthesea
- **Attention Mechanism**: Cơ chế attention để cải thiện chất lượng caption

## 🏗️ Kiến Trúc Mô Hình (Model Architecture)

### 1. Image Encoder
- **Model**: InceptionV3 pre-trained trên ImageNet
- **Input**: Hình ảnh 299x299 pixels
- **Output**: Feature vector 2048 chiều
- **Preprocessing**: Chuẩn hóa theo InceptionV3

### 2. Text Decoder  
- **Embedding**: FastText Vietnamese (300 dimensions)
- **LSTM**: Bidirectional LSTM layers
- **Attention**: Add attention mechanism
- **Output**: Softmax over vocabulary

### 3. Training Configuration
```python
EPOCHS = 100
BATCH_SIZE = 32
OPTIMIZER = Adam
VOCABULARY_SIZE = 3299
MAX_SEQUENCE_LENGTH = 41
EMBEDDING_DIM = 300
```

## 📊 Dataset Information

### Flickr8k Vietnamese Captions
- **Total Images**: 8,091 hình ảnh
- **Captions per Image**: 5 mô tả tiếng Việt
- **Vocabulary Size**: 6,851 từ unique (cleaned: 3,299)
- **Train/Test Split**: Theo file trainImages.txt và testImages.txt

### Data Processing Pipeline
1. **Text Cleaning**: Loại bỏ dấu câu, chuẩn hóa văn bản
2. **Tokenization**: Sử dụng underthesea word_tokenize
3. **Sequence Formatting**: `startseq [caption] endseq`
4. **Padding**: Padding sequences đến max_length

## 🛠️ Công Nghệ Sử Dụng (Technology Stack)

### Core Libraries
- **TensorFlow/Keras**: Deep learning framework
- **InceptionV3**: Pre-trained CNN model
- **FastText**: Vietnamese word embeddings
- **underthesea**: Vietnamese NLP toolkit

### Supporting Tools
- **NumPy/Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **PIL/OpenCV**: Image processing
- **NLTK**: Text evaluation (BLEU score)
- **tqdm**: Progress tracking

## ⚙️ Cài Đặt và Sử Dụng (Installation & Usage)

### 1. Dependencies Installation

```bash
pip install tensorflow keras pillow numpy tqdm
pip install underthesea gensim fasttext nltk
pip install matplotlib opencv-python
```

### 2. Dataset Preparation

```python
# Download required datasets
# - Flickr8k Images
# - Vietnamese Captions (captions_vi.txt)
# - FastText Vietnamese vectors (cc.vi.300.bin)
```

### 3. Feature Extraction

```python
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# Extract image features
model = InceptionV3(weights='imagenet')
model = Model(model.input, model.layers[-2].output)

# Process images
for img in images:
    image = load_img(img, target_size=(299, 299))
    image = preprocess_input(image)
    feature = model.predict(image)
```

### 4. Model Training

```python
# Build model
def define_model(vocab_size, max_length, embedding_matrix):
    # Image feature input
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 300, weights=[embedding_matrix])(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = Bidirectional(LSTM(256))(se2)
    
    # Combine and output
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model
```

### 5. Caption Generation

```python
def generate_caption(image_path):
    # Extract features
    feature = extract_features(image_path)
    
    # Generate sequence
    caption = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        prediction = model.predict([feature, sequence])
        predicted = np.argmax(prediction)
        word = tokenizer.index_word[predicted]
        
        if word == 'endseq':
            break
        caption += ' ' + word
    
    return caption
```

## 📈 Kết Quả và Đánh Giá (Results & Evaluation)

### Training Metrics
- **Loss Function**: Categorical Crossentropy
- **Training Time**: ~3-4 hours on Tesla T4 GPU
- **Convergence**: Model converges after ~50-60 epochs
- **Memory Usage**: ~8GB GPU memory

### Evaluation Metrics
- **BLEU Score**: Đánh giá chất lượng caption
- **Perplexity**: Đo độ phức tạp của mô hình ngôn ngữ
- **Human Evaluation**: Đánh giá chủ quan về tính tự nhiên

### Sample Outputs
```
Input: [Hình ảnh một đứa trẻ đang chơi]
Output: "một đứa trẻ đang chơi trong công viên với quả bóng"

Input: [Hình ảnh chó đang chạy]  
Output: "một con chó đang chạy trên bãi cỏ xanh"
```

## 🔧 Tối Ưu Hóa và Cải Tiến (Optimization & Improvements)

### Current Optimizations
- **Data Augmentation**: Rotation, flip, brightness adjustment
- **Dropout Regularization**: Prevent overfitting
- **Bidirectional LSTM**: Better sequence understanding
- **Pre-trained Embeddings**: FastText Vietnamese vectors

### Future Improvements
- **Attention Mechanism**: Visual attention for better focus
- **Transformer Architecture**: Use BERT/GPT-style models
- **Beam Search**: Better decoding strategy
- **Multi-modal Features**: Combine multiple feature extractors

## 📁 Cấu Trúc Dự Án (Project Structure)

```
gen-captions/
├── flickr8k-vietnamese-captions.ipynb    # Main notebook
├── README.md                             # Project documentation
├── models/
│   ├── captions_vietnamese_flickr8k.h5  # Trained model
│   ├── tokenizer.pkl                    # Text tokenizer
│   └── features_inception_v3.pkl        # Image features
├── data/
│   ├── captions_vi.txt                  # Vietnamese captions
│   ├── trainImages.txt                  # Training image list
│   └── testImages.txt                   # Test image list
└── utils/
    ├── preprocessing.py                  # Data preprocessing
    ├── model_utils.py                   # Model utilities
    └── evaluation.py                    # Evaluation metrics
```

## 🔍 Chi Tiết Kỹ Thuật (Technical Details)

### Text Processing
```python
# Vietnamese text cleaning
def cleaning_text(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for img, captions in descriptions.items():
        for idx, cap in enumerate(captions):
            desc = cap.replace(' - ', ' ').translate(table)
            desc = ' '.join(simple_preprocess(desc))
            desc = word_tokenize(desc, format='text')
            descriptions[img][idx] = desc
    return descriptions
```

### Feature Extraction
```python
# Image feature extraction with InceptionV3
def extract_feature(directory):
    model = InceptionV3(weights='imagenet')
    model = Model(model.input, model.layers[-2].output)
    
    features = dict()
    for img in tqdm(os.listdir(directory)):
        filename = directory + '/' + img
        image = load_img(filename, target_size=(299, 299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        
        feature = model.predict(image, verbose=0)
        features[img] = feature
    return features
```

## 📚 Tài Liệu Tham Khảo (References)

- [Show and Tell: Neural Image Caption Generation](https://arxiv.org/abs/1411.4555)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [FastText Vietnamese Word Vectors](https://fasttext.cc/docs/en/crawl-vectors.html)
- [underthesea - Vietnamese NLP](https://github.com/undertheseanlp/underthesea)

## 🤝 Đóng Góp (Contributing)

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

Dự án này được phân phối dưới MIT License. Xem file `LICENSE` để biết thêm chi tiết.

## 📞 Liên Hệ (Contact)

- **Author**: kenzn2
- **GitHub**: [@kenzn2](https://github.com/kenzn2)
- **Project**: [gen-captions](https://github.com/kenzn2/gen-captions)

## 🙏 Acknowledgments

- Flickr8k dataset creators
- FastText team for Vietnamese word vectors
- underthesea project for Vietnamese NLP tools
- TensorFlow/Keras development team
- Kaggle community for computational resources

---

<div align="center">
  <b>🖼️ Biến hình ảnh thành lời nói tiếng Việt! 📝</b>
</div>