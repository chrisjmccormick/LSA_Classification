This is a simple text classification example using Latent Semantic Analysis 
(LSA), written in Python and using the scikit-learn library. 

LSA is also referred to as Latent Semantic Indexing (LSI).



Scripts
-------
<table>
<tr><td>getReutersTextArticles.py</td><td>Pulls down the raw text dataset to experiment with. The dataset is then written to data/raw_text_dataset.pickle</td></tr>
<tr><td>runClassification_LSA.py</td><td>For a baseline accuracy comparison, uses LSA to vectorize the document, then runs k-NN classification.</td></tr>
</table>

