Download Link: https://assignmentchef.com/product/solved-ece219-project-1-classification-analysis-on-textual-data
<br>
Statistical classification refers to the task of identifying a category, from a predefined set, to which a data point belongs, given a training data set with known category memberships. Classification differs from the task of clustering, which concerns grouping data points with no predefined category memberships, where the objective is to seek inherent structures in data with respect to suitable measures. Classification turns out as an essential element of data analysis, especially when dealing with a large amount of data. In this project, we look into different methods for classifying textual data.

In this project, the goal includes:

<ol>

 <li>To learn how to construct tf-idf representations of textual data.</li>

 <li>To get familiar with various common classification methods.</li>

 <li>To learn ways to evaluate and diagnose classification results.</li>

 <li>To learn two dimensionality reduction methods: PCA &amp; NMF.</li>

 <li>To get familiar with the complete pipeline of a textual data classification task.</li>

</ol>

<h2>Getting familiar with the dataset</h2>

We work with “20 Newsgroups” dataset, which is a collection of approximately 20,000 documents, partitioned (nearly) evenly across 20 different newsgroups (newsgroups are discussion groups like forums, which originated during the early age of the Internet), each corresponding to a different topic.

One can use fetch 20newsgroups provided by scikit-learn to load the dataset. Detailed usages can be found at <a href="https://scikit-learn.org/stable/datasets/#the-20-newsgroups-text-dataset">https://scikit-learn.org/stable/datasets/#the-20-newsgroups-text-dataset</a>

In a classification problem one should make sure to properly handle any imbalance in the relative sizes of the data sets corresponding to different classes. To do so, one can either modify the penalty function (<em>i.e. </em>assign more weight to errors from minority classes), or alternatively, down-sample the majority classes, to have the same number of instances as minority classes.

<strong>QUESTION 1: </strong>To get started, plot a histogram of the number of training documents for each of the 20 categories to check if they are evenly distributed.

Note that the data set is already balanced (especially for the categories we’ll mainly work on) and so in this case we do not need to balance. But in general, as a data scientist you need to be aware of this issue.

<h2>Binary Classification</h2>

Before the following parts, to ensure consistency, please set the random seed as follows:

import numpy as np np.random.seed(42) import random random.seed(42)

To get started, we work with a well separable portion of data, and see if we can train a classifier that distinguishes two classes well. Concretely, let us take all the documents in the following classes:

Table 1: Two well-separated classes

<table width="691">

 <tbody>

  <tr>

   <td width="127">Computer Technology</td>

   <td width="564">comp.graphics comp.os.ms-windows.misc comp.sys.ibm.pc.hardware comp.sys.mac.hardware</td>

  </tr>

  <tr>

   <td width="127">Recreational Activity</td>

   <td width="564">rec.autos                       rec.motorcycles                             rec.sport.baseball                          rec.sport.hockey</td>

  </tr>

 </tbody>

</table>

Specifically, use the settings as the following code to load the data:

<table width="643">

 <tbody>

  <tr>

   <td width="643">categories = [‘comp.graphics’, ‘comp.os.ms-windows.misc’,‘comp.sys.ibm.pc.hardware’, ‘comp.sys.mac.hardware’,‘rec.autos’, ‘rec.motorcycles’,‘rec.sport.baseball’, ‘rec.sport.hockey’]train_dataset = fetch_20newsgroups(subset = ‘train’, categories = categories,<em>,</em><sub>→ </sub>shuffle = True, random_state = None) test_dataset = fetch_20newsgroups(subset = ‘test’, categories = categories,<em>,</em><sub>→ </sub>shuffle = True, random_state = None)</td>

  </tr>

 </tbody>

</table>

<h3>1      Feature Extraction</h3>

The primary step in classifying a corpus of text is choosing a proper document representation. A good representation should retain enough information that enable us to perform the classification, yet in the meantime, be concise to avoid computational intractability and over fitting.

One common representation of documents is called “Bag of Words”, where a document is represented as a histogram of term frequencies, or other statistics of the terms, within a fixed vocabulary. As such, a corpus of text can be summarized into a term-document matrix whose entries are some statistic of the terms.

First a common sense filtering is done to drop certain words or terms: to avoid unnecessarily large feature vectors (vocabulary size), terms that are too frequent in almost every document, or are very rare, are dropped out of the vocabulary. The same goes with special characters, common stop words (e.g. “and”, “the” etc.), In addition, appearances of words that share the same stem in the vocabulary (e.g. “goes” vs “going”) are merged into a single term.

Further, one can consider using the normalized count of the vocabulary words in each document to build representation vectors. A popular numerical statistic to capture the importance of a word to a document in a corpus is the “Term Frequency-Inverse Document Frequency (TF-IDF)” metric. This measure takes into account count of the words in the document, as normalized by a certain function of the frequency of the individual words in the whole corpus. For example, if a corpus is about computer accessories then words such as “computer” “software” “purchase” will be present in almost every document and their frequency is not a distinguishing feature for any document in the corpus. The discriminating words will most likely be those that are specialized terms describing different types of accessories and hence will occur in fewer documents. Thus, a human reading a particular document will usually ignore the contextually dominant words such as “computer”, “software” etc. and give more importance to specific words. This is like when going into a very bright room or looking at a bright object, the human perception system usually applies a saturating function (such as a logarithm or square-root) to the actual input values before passing it on to the neurons. This makes sure that a contextually dominant signal does not overwhelm the decision-making processes in the brain. The TF-IDF functions draw their inspiration from such neuronal systems. Here we define the TF-IDF score to be

tf-idf(<em>d,t</em>) = tf(<em>t,d</em>) × idf(<em>t</em>)

where tf(<em>d,t</em>) represents the frequency of term <em>t </em>in document <em>d</em>, and inverse document frequency is defined as:

idf(

where <em>n </em>is the total number of documents, and df(<em>t</em>) is the document frequency, <em>i.e. </em>the number of documents that contain the term <em>t</em>.

<strong>QUESTION 2: </strong>Use the following specs to extract features from the textual data:

<ul>

 <li>Use the “english” stopwords of the CountVectorizer</li>

 <li>Exclude terms that are numbers (e.g. “123”, “-45”, “6.7” etc.)</li>

 <li>Perform lemmatization with nltk.wordnet.WordNetLemmatizer and pos tag</li>

 <li>Use min df=3</li>

</ul>

Report the shape of the TF-IDF matrices of the train and test subsets respectively.

Please refer to <a href="http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html">the official documentation of</a> <a href="http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html">CountVectorizer</a> as well as the discussion section notebooks for details.

<h3>2      Dimensionality Reduction</h3>

After above operations, the dimensionality of the representation vectors (TF-IDF vectors) ranges in the order of thousands. However, learning algorithms may perform poorly in highdimensional data, which is sometimes referred to as “The Curse of Dimensionality”. Since the document-term TF-IDF matrix is sparse and low-rank, as a remedy, one can select a subset of the original features, which are more relevant with respect to certain performance measure, or transform the features into a lower dimensional space.

In this project, we use two dimensionality reduction methods: Latent Semantic Indexing (LSI) and Non-negative Matrix Factorization (NMF), both of which minimize mean squared residual between the original data and a reconstruction from its low-dimensional approximation. Recall that our data is the term-document TF-IDF matrix, whose rows correspond to TF-IDF representation of the documents, <em>i.e.</em>

<table width="221">

 <tbody>

  <tr>

   <td width="99">tfidf(<em>d</em><sub>1</sub><em>,t</em><sub>1</sub>) tfidf(<em>d ,t </em>)</td>

   <td width="32">······</td>

   <td width="90">tfidf(<em>d</em><sub>1</sub><em>,t<sub>m</sub></em>) tfidf(<em>d ,t </em>)</td>

  </tr>

 </tbody>

</table>

<sub>2                       1                     2                     <em>m </em></sub><sub> </sub><strong>X</strong>… …          .            

<sub></sub>

tfidf(<em>d<sub>n</sub>,t</em><sub>1</sub>) ···      tfidf(

(which is the case for the output of CountVectorizer and TfidfTransformer).

<h4>LSI</h4>

The LSI representation is obtained by computing left and right singular vectors corresponding to the top <em>k </em>largest singular values of the term-document TF-IDF matrix <strong>X</strong>.

We perform SVD to the matrix <strong>X</strong>, resulting in <strong>X </strong>= <strong>UΣV</strong><sup>T</sup>, <strong>U </strong>and <strong>V </strong>orthogonal. Let the singular values in <strong>Σ </strong>be sorted in descending order, then the first <em>k </em>columns of <strong>U </strong>and <strong>V </strong>are called <strong>U</strong><em><sub>k </sub></em>and <strong>V</strong><em><sub>k </sub></em>respectively. <strong>V</strong><em><sub>k </sub></em>consists of the principle components in the feature space.

Then we use (<strong>XV</strong><em><sub>k</sub></em>) (which is also equal to (<strong>U</strong><em><sub>k</sub></em><strong>Σ</strong><em><sub>k</sub></em>)) as the dimension-reduced data matrix, where rows still correspond to documents, only that they can have (far) lower dimension. In this way, the number of features is reduced. LSI is similar to Principal Component Analysis (PCA), and you can see the lecture notes for their relationships.

Having learnt <strong>U </strong>and <strong>V</strong>, to reduce the <em>test </em>data, we just multiply the test TF-IDF matrix <strong>X</strong><em><sub>t </sub></em>by <strong>V</strong><em><sub>k</sub></em>, <em>i.e. </em><strong>X</strong><em><sub>t,</sub></em><sub>reduced </sub>= <strong>X</strong><em><sub>t</sub></em><strong>V</strong><em><sub>k</sub></em>. By doing so, we actually project the test TF-IDF vectors to the principle components, and use the projections as the dimension-reduced data.

<h4>NMF</h4>

NMF tries to approximate the data matrix <strong>X </strong>∈ R<em><sup>n</sup></em><sup>×<em>m </em></sup>(<em>i.e. </em>we have <em>n </em>docs and <em>m </em>terms) with <strong>WH </strong>(<strong>W </strong>∈ R<em><sup>n</sup></em><sup>×<em>r</em></sup>, <strong>H </strong>∈ R<em><sup>r</sup></em><sup>×<em>m</em></sup>). Concretely, it finds the non-negative matrices <strong>W </strong>and <strong>H </strong>s.t.

k<strong>X </strong>− <strong>WH</strong>k<sup>2</sup><em><sub>F </sub></em>is minimized. () Then we use <strong>W </strong>as the dim-reduced data

matrix, and in the fit step, we calculate both <strong>W </strong>and <strong>H</strong>. The intuition behind this is that we are trying to describe the documents (the rows in <strong>X</strong>) as a (non-negative) linear combination of <em>r </em>topics:

<h3>X  WH</h3>

Here we see <strong>h</strong> “topics”, each of which consists of <em>m </em>scores, indicating how important each term is in the topic. Then <strong>x</strong>.

Now how do we calculate the dim-reduced test data matrix? Again, we try to describe the document vectors (rows by our convention here) in the test data (call it <strong>X</strong><em><sub>t</sub></em>) with (non-negative) linear combinations of the “topics” we learned in the fit step. The “topics”, again, are the rows of <strong>H </strong>matrix,. How do we do that? Just solve the optimization problem

where <strong>H </strong>is fixed as the <strong>H </strong>matrix we learned in the fit step. Then <strong>W</strong><em><sub>t </sub></em>is used as the dim-reduced version of <strong>X</strong><em><sub>t</sub></em>.

<strong>QUESTION 3: </strong>Reduce the dimensionality of the data using the methods above

<ul>

 <li>Apply LSI to the TF-IDF matrix corresponding to the 8 categories with <em>k </em>= 50; so each document is mapped to a 50-dimensional vector.</li>

 <li>Also reduce dimensionality through NMF (<em>k </em>= 50) and compare with LSI:</li>

</ul>

Which one is larger, the k<strong>X </strong>− <strong>WH</strong>k<sup>2</sup><em><sub>F </sub></em>in NMF or the  in LSI?

Why is the case?

<h3>3       Classification Algorithms</h3>

<strong>In this part, you are asked to use the dimension-reduced training data from LSI to train (different types of) classifiers, and evaluate the trained classifiers with test data. </strong>Your task would be to classify the documents into two classes “Computer Technology” vs “Recreational Activity”. Refer to Table 1 to find the 4 categories of documents comprising each of the two classes. In other words, you need to combine documents of those sub-classes of each class to form the set of documents for each class.

<h4>Classification measures</h4>

Classification quality can be evaluated using different measures such as <strong>precision</strong>, <strong>recall</strong>, <strong>F-score</strong>, etc. Refer to the discussion material to find their definition.

Depending on application, the true positive rate (TPR) and the false positive rate (FPR) have different levels of significance. In order to characterize the trade-off between the two quantities, we plot the receiver operating characteristic (ROC) curve. For binary classification, the curve is created by plotting the true positive rate against the false positive rate at various threshold settings on the probabilities assigned to each class (let us assume probability <em>p </em>for class 0 and 1 − <em>p </em>for class 1). In particular, a threshold <em>t </em>is applied to value of <em>p </em>to select between the two classes. The value of threshold <em>t </em>is swept from 0 to 1, and a pair of TPR and FPR is got for each value of <em>t</em>. The ROC is the curve of TPR plotted against FPR.

<h4>SVM</h4>

Linear Support Vector Machines have been proved efficient when dealing with sparse high dimensional datasets, including textual data. They have been shown to have good generalization accuracy, while having low computational complexity.

Linear Support Vector Machines aim to learn a vector of feature weights, <strong>w</strong>, and an intercept, <em>b</em>, given the training dataset. Once the weights are learned, the label of a data point is determined by thresholding <strong>w</strong><sup>T</sup><strong>x </strong>+ <em>b </em>with 0, <em>i.e. </em>s<em>ign</em>(<strong>w</strong><sup>T</sup><strong>x </strong>+ <em>b</em>). Alternatively, one produce probabilities that the data point belongs to either class, by applying a logistic function instead of hard thresholding, <em>i.e. </em>calculating <em>σ</em>(<strong>w</strong><sup>T</sup><strong>x </strong>+ <em>b</em>).

The learning process of the parameter <strong>w </strong>and <em>b </em>involves solving the following optimization problem:

<em>n</em>

<em>s.t. y<sub>i</sub></em>(<strong>w</strong><sup>T</sup><strong>x</strong><em><sub>i </sub></em>+ <em>b</em>) ≥ 1 − <em>ξ<sub>i </sub>ξ<sub>i </sub></em>≥ 0<em>, </em>∀<em>i </em>∈ {1<em>,…,n</em>}

where <strong>x</strong><em><sub>i </sub></em>is the <em>i</em>th data point, and <em>y<sub>i </sub></em>∈ {0<em>,</em>1} is the class label of it.

Minimizing the sum of the slack variables corresponds to minimizing the loss function on the training data. On the other hand, minimizing the first term, which is basically a regularization, corresponds to maximizing the margin between the two classes. Note that in the objective function, each slack variable represents the amount of error that the classifier can tolerate for a given data sample. The tradeoff parameter <em>γ </em>controls relative importance of the two components of the objective function. For instance, when 1, misclassification of individual points is highly penalized, which is called “Hard Margin SVM”. In contrast, a “Soft Margin SVM”, which is the case when 1, is very lenient towards misclassification of a few individual points as long as most data points are well separated.

<strong>QUESTION 4: </strong>Hard margin and soft margin linear SVMs:

<ul>

 <li>Train two linear SVMs and compare:

  <ul>

   <li>Train one SVM with <em>γ </em>= 1000 (hard margin), another with <em>γ </em>= 0<em>.</em>0001 (soft margin).</li>

   <li>Plot the ROC curve, report the confusion matrix and calculate the accuracy, recall, precision and F-1 score of both SVM classifier. Which one performs better?</li>

   <li>What happens for the soft margin SVM? Why is the case?</li>

  </ul></li>

</ul>

∗ Does the ROC curve of the soft margin SVM look good? Does this conflict with other metrics?

<ul>

 <li>Use cross-validation to choose <em>γ </em>(use average validation accuracy to compare):</li>

</ul>

Using a 5-fold cross-validation, find the best value of the parameter <em>γ </em>in the range {10<em><sup>k</sup></em>|−3 ≤ <em>k </em>≤ 3<em>,k </em>∈ Z}. Again, plot the ROC curve and report the confusion matrix and calculate the accuracy, recall precision and F-1 score of this best SVM.

<h4>Logistic Regression</h4>

Although its name contains “regression”, logistic regression is a probability model that is used for binary classification.

In logistic regression, a logistic function (<em>σ</em>(<em>φ</em>) = 1<em>/</em>(1+exp(−<em>φ</em>))) acting on a linear function of the features (<em>φ</em>(<strong>x</strong>) = <strong>w</strong><sup>T</sup><strong>x</strong>+<em>b</em>) is used to calculate the probability that the data point belongs to class 1, and during the training process, <strong>w </strong>and <em>b </em>that maximizes the likelihood of the training data are learnt.

One can also add regularization term in the objective function, so that the goal of the training process is not only maximizing the likelihood, but also minimizing the regularization term, which is often some norm of the parameter vector <strong>w</strong>. Adding regularization helps prevent ill-conditioned results and over-fitting, and facilitate generalization ability of the classifier. A coefficient is used to control the trade-off between maximizing likelihood and minimizing the regularization term.

<strong>QUESTION 5: </strong>Logistic classifier:

<ul>

 <li>Train a logistic classifier without regularization (you may need to come up with some way to approximate this if you use sklearn.linear model.LogisticRegression); plot the ROC curve and report the confusion matrix and calculate the accuracy, recall precision and F-1 score of this classifier.</li>

 <li>Regularization:

  <ul>

   <li>Using 5-fold cross-validation on the dimension-reduced-by-svd training data, find the best regularization strength in the range {10<em><sup>k</sup></em>|−3 ≤ <em>k </em>≤ 3<em>,k </em>∈ Z} for logistic regression with L1 regularization and logistic regression L2 regularization, respectively.</li>

   <li>Compare the performance (accuracy, precision, recall and F-1 score) of 3 logistic classifiers: w/o regularization, w/ L1 regularization and w/ L2 regularization (with the best parameters you found from the part above), using test data.</li>

   <li>How does the regularization parameter affect the test error? How are the learnt coefficients affected? Why might one be interested in each type of regularization?</li>

   <li>Both logistic regression and linear SVM are trying to classify data points using a linear decision boundary, then what’s the difference between their ways to find this boundary? Why their performance differ?</li>

  </ul></li>

</ul>

<h4>Na¨ıve Bayes</h4>

Scikit-learn provides a type of classifiers called “na¨ıve Bayes classifiers”. They include MultinomialNB, BernoulliNB, and GaussianNB.

Na¨ıve Bayes classifiers use the assumption that features are statistically independent of each other when conditioned by the class the data point belongs to, to simplify the calculation for the Maximum A Posteriori (MAP) estimation of the labels. That is,

<em>P</em>(<em>x<sub>i </sub></em>| <em>y,x</em><sub>1</sub><em>,…,x<sub>i</sub></em><sub>−1</sub><em>,x<sub>i</sub></em><sub>+1</sub><em>,…,x<sub>m</sub></em>) = <em>P</em>(<em>x<sub>i </sub></em>| <em>y</em>)                      <em>i </em>∈ {1<em>,…,m</em>}

where <em>x<sub>i</sub></em>’s are features, <em>i.e. </em>components of a data point, and <em>y </em>is the label of the data point.

Now that we have this assumption, a probabilistic model is still needed; the difference between MultinomialNB, BernoulliNB, and GaussianNB is that they use different models.

<strong>QUESTION 6: </strong>Na¨ıve Bayes classifier: train a GaussianNB classifier; plot the ROC curve and report the confusion matrix and calculate the accuracy, recall, precision and F-1 score of this classifier.

<h3>Grid Search of Parameters</h3>

Now we have gone through the complete process of training and testing a classifier. However, there are lots of parameters that we can tune. In this part, we fine-tune the parameters.

<strong>QUESTION 7: </strong>Grid search of parameters:

<ul>

 <li>Construct a Pipeline that performs feature extraction, dimensionality reduction and classification;</li>

 <li>Do grid search with 5-fold cross-validation to compare the following (use test accuracy as the score to compare):</li>

</ul>

Table 2: Options to compare

<table width="558">

 <tbody>

  <tr>

   <td width="173"><strong>Procedure</strong></td>

   <td width="385"><strong>Options</strong></td>

  </tr>

  <tr>

   <td width="173">Loading Data</td>

   <td width="385">remove “headers” and “footers” vs not</td>

  </tr>

  <tr>

   <td width="173">Feature Extraction</td>

   <td width="385">min df = 3 vs 5;use lemmatization vs not</td>

  </tr>

  <tr>

   <td width="173">Dimensionality Reduction</td>

   <td width="385">LSI vs NMF</td>

  </tr>

  <tr>

   <td width="173">Classifier</td>

   <td width="385">SVM with the best <em>γ </em>previously foundvsLogistic Regression: L1 regularization vs L2 regularization, with the best regularization strength previously found vsGaussianNB</td>

  </tr>

  <tr>

   <td width="173">Other options</td>

   <td width="385">Use default</td>

  </tr>

 </tbody>

</table>

<ul>

 <li>What is the best combination?</li>

</ul>

Hint: see

<a href="http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html">http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html </a>and <a href="http://www.davidsbatista.net/blog/2018/02/23/model_optimization/">http://www.davidsbatista.net/blog/2018/02/23/model_optimization/</a>

<h2>Multiclass Classification</h2>

So far, we have been dealing with classifying the data points into two classes. In this part, we explore multiclass classification techniques through different algorithms.

Some classifiers perform the multiclass classification inherently. As such, na¨ıve Bayes algorithm finds the class with maximum likelihood given the data, regardless of the number of classes. In fact, the probability of each class label is computed in the usual way, then the class with the highest probability is picked; that is

<em>c</em>ˆ= argmin<em>P</em>(<em>c </em>| <strong>x</strong>)

<em>c</em>∈C

where <em>c </em>denotes a class to be chosen, and ˆ<em>c </em>denotes the optimal class.

For SVM, however, one needs to extend the binary classification techniques when there are multiple classes. A natural way to do so is to perform a one versus one classification on all

pairs of classes, and given a document the class is assigned with the majority vote.

In case there is more than one class with the highest vote, the class with the highest total classification confidence levels in the binary classifiers is picked.

An alternative strategy would be to fit one classifier per class, which reduces the number of classifiers to be learnt to |C|. For each classifier, the class is fitted against all the other classes. Note that in this case, the unbalanced number of documents in each class should be handled. By learning a single classifier for each class, one can get insights on the interpretation of the classes based on the features.

<strong>QUESTION 8: </strong>In this part, we aim to learn classifiers on the documents belonging to the classes:

comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, misc.forsale, soc.religion.christian

Perform Na¨ıve Bayes classification and multiclass SVM classification (with both One VS One and One VS the rest methods described above) and report the confusion matrix and calculate the accuracy, recall, precision and F-1 score of your classifiers.