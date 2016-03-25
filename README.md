This is a simple experiment performed to evaluate the effectiveness of the
Cortical.io Retina NLP algorithm. In this case, I'm applying it to a small
text classification exercise, and comparing the accuracy of the results to
Latent Semantic Analysis (LSA).

Scripts
-------
<table>
<tr><td>getReutersTextArticles.py</td><td>Pulls down the raw text dataset to experiment with. The dataset is then written to data/raw_text_dataset.pickle</td></tr>
<tr><td>getFingerprints.py</td><td>Uses the REST interface to the Cortical.io Retina API to calculate fingerprints for the text articles. The fingerprinted dataset is then written to data/fingerprint_dataset.pickle</td></tr>
<tr><td>runClassification_Fingerprints.py</td><td>Runs k-NN classification on the fingerprinted dataset.</td></tr>
<tr><td>runClassification_LSA.py</td><td>For a baseline accuracy comparison, uses LSA to vectorize the document, then runs k-NN classification.</td></tr>
</table>

