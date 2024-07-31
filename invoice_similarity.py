import fitz
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np

# Text Extraction
def extract_text_from_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Text Preprocessing
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    return text

# Extracting Features
def extract_features(texts):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

# Finding Similarities
def find_most_similar_invoice(input_text, existing_texts):
    all_texts = existing_texts + [input_text]
    features, vectorizer = extract_features(all_texts)
    similarity_matrix = cosine_similarity(features)
    similarities = similarity_matrix[-1, :-1]
    most_similar_index = similarities.argmax()
    return most_similar_index, similarities[most_similar_index]

# Load invoices
def load_invoices(folder_path):
    texts = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(file_path)
            preprocessed_text = preprocess_text(text)
            texts.append(preprocessed_text)
            filenames.append(filename)
    return texts, filenames

# Extract logos from PDF
def extract_logos_from_pdf(pdf_path):
    logos = []
    pdf_document = fitz.open(pdf_path)
    for page_index in range(len(pdf_document)):
        page = pdf_document.load_page(page_index)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                logos.append(image)
    return logos

# Compute logo similarity
def compute_logo_similarity(test_descriptors, train_descriptors):
    if not (test_descriptors is not None and len(test_descriptors) > 0):
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = []
    for test_desc in test_descriptors:
        if test_desc is not None:
            for train_desc in train_descriptors:
                if train_desc is not None:
                    # Ensure descriptors are in the same format
                    if test_desc.dtype != train_desc.dtype:
                        continue
                    # Find matches
                    matches.extend(bf.match(test_desc, train_desc))
    
    if len(matches) == 0:
        return 0
    return len(matches)

# Extract ORB features from image
def extract_orb_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

# Compare logos
def compare_logos(test_logos, train_logos_list):
    results = []
    for test_logo in test_logos:
        test_descriptors = extract_orb_features(test_logo)
        if test_descriptors is None:
            results.append(0)
            continue

        max_similarity = 0
        for train_logos in train_logos_list:
            train_descriptors = []
            for logo in train_logos:
                descriptors = extract_orb_features(logo)
                if descriptors is not None:
                    train_descriptors.append(descriptors)

            # Compare with each logo's descriptors in train
            for train_desc in train_descriptors:
                similarity = compute_logo_similarity(test_descriptors, train_desc)
                max_similarity = max(max_similarity, similarity)
        results.append(max_similarity)
    
    return results

# Main function
def main():
    train_folder = 'train'
    test_folder = 'test'

    # Load training and test texts
    train_texts, train_filenames = load_invoices(train_folder)
    test_texts, test_filenames = load_invoices(test_folder)

    # Compare text
    for test_filename, test_text in zip(test_filenames, test_texts):
        most_similar_index, similarity_score = find_most_similar_invoice(test_text, train_texts)
        print(f'Test PDF {test_filename}:')
        print(f'  Most similar training PDF based on text: {train_filenames[most_similar_index]}')
        print(f'  Similarity score: {similarity_score:.4f}')

    # Extract logos from PDFs
    train_logos_list = [extract_logos_from_pdf(os.path.join(train_folder, filename)) for filename in train_filenames]
    test_logos_list = [extract_logos_from_pdf(os.path.join(test_folder, filename)) for filename in test_filenames]

    # Compare logos
    for i, test_logos in enumerate(test_logos_list):
        if len(test_logos) == 0:
            print(f'Test PDF {test_filenames[i]}: No logos found.')
            continue

        logo_similarities = compare_logos(test_logos, train_logos_list)
        if len(logo_similarities) == 0:
            print(f'Test PDF {test_filenames[i]}: No logo similarities found.')
            continue

        max_similarity_index = logo_similarities.index(max(logo_similarities))
        print(f'Test PDF {test_filenames[i]}:')
        print(f'  Most similar training PDF based on logos: {train_filenames[max_similarity_index]}')
        print(f'  Similarity score: {logo_similarities[max_similarity_index]:.4f}')

if __name__ == "__main__":
    main()
