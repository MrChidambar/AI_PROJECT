Invoice Similarity Detection:
Overview
This project identifies the most similar PDF document from a training set (train folder) to a given test PDF (test folder) based on textual and structural features. The similarity is determined using a combination of Jaccard Similarity for initial filtering and Cosine Similarity for detailed comparison of document styles and content.
Document Representation
![WhatsApp Image 2024-07-31 at 22 51 11_9b9fa90f](https://github.com/user-attachments/assets/1934533b-990b-4d55-80e5-1847cdf752a5)

1.	Text Representation:
Text is extracted from PDFs using OCR via the pyPDF2 library.
The extracted text is used to calculate similarity metrics.
2.	Structural Features:
	Additional document features such as footer, header, table, and image styles are extracted and compared.
Similarity Metrics
1.	Jaccard Similarity:
Used for an initial comparison of the extracted text from the test PDF against the text in the training PDFs. Jaccard Similarity is calculated based on the presence of common words.
2.	Cosine Similarity:
	Applied to compare specific structural features (footer, header, table, image styles) of the test PDF with those in the training PDFs. Cosine Similarity measures the cosine of the angle between two vectors in a multi-dimensional space, indicating how similar the vectors (documents) are.
Instructions to Run the Code
Prerequisites
•	Python 3.x
•	PyMuPDF (for PDF text extraction)
•	scikit-learn (for similarity metrics)
•	opencv-python (for image processing)
•	numpy (for numerical operations)
•	pyPDF2 (for additional PDF processing)
1.	Install Required Packages
pip install PyMuPDF scikit-learn opencv-python numpy pyPDF2
2.	Data Preparation
	Place the test PDF(s) in the test folder.
	Place the training PDFs in the train folder.
Running the Code
1.	Run the Similarity Detection Script
Copy code
python invoice_similarity.py
2.	Output
	The script will output the most similar training PDF for each test PDF, along with the similarity score.
Example Output

Test PDF: test.pdf
  Most similar training PDF based on text: train2.pdf
  Similarity score: 0.7919

Test PDF: test.pdf
  Most similar training PDF based on structural features: train1.pdf
  Similarity score: 0.8856
Approach Breakdown
1.	Text Extraction and Initial Filtering:
	Text from each test PDF is extracted and compared against the text from the training PDFs using Jaccard Similarity. This step quickly filters out dissimilar documents.
2.	Structural Feature Extraction:
	For documents passing the initial filter, additional features like footer, header, tables, and image styles are extracted.
3.	Detailed Comparison:
	Cosine Similarity is used to compare the structural features between the test PDF and training PDFs. The training PDF with the highest similarity score is deemed most similar.

