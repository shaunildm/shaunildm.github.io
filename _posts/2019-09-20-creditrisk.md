---
title: "Credit Risk Modeling"
date: 2019-09-20
tags: [Prediction Models]
header:
  image: "/images/credit-risk/p2p.jpg"
excerpt: "Data Science, Credit Risk, Machine Learning"
---

# Credit Risk Modelling Of Peer-to-Peer Lending Services

Peer-to-peer lending is a newer form of lending which connects borrowers to lenders through various online marketplaces. This cuts out fees, protocols and other costs normally incurred by traditonal banks. Peer-to-peer lending is expected to grow very quickly, a report from [Transparency Market Research (TMR)](https://www.transparencymarketresearch.com/report-toc/10835) estimates a compound annual growth rate of 48.2% by 2024.

## Focus

We want to create a model that can predict whether a loan will be paid off on-time or not. The model needs to be built with simplicity with mind so it's workflow can quickly be adapted to growing and changing data.

## Data

We will use data from the peer-to-peer lending service, [Lending Club](https://www.lendingclub.com). This is ideal considering p2p lending is relatively new and we need data from at least a few years back which Lending Club has since it has been around for a while and they serve a lot customers.


Each prospective borrower applies by providing financial history, the loans purpose, and other credit relevant data and Lending Club assigns an interest rate and grade to the loan or rejects the loan by using their own models. A loan with a higher interest rate with generate a higher return with a higher risk. Conversly, a loan with a lower interest will have lower risk and lower return. As conservative investors, we would like to create a model that can provide us with accurate and relevant predictions.


## Understanding the Data

Lending Club has loan data available on there [website](https://www.lendingclub.com/info/download-data.action), which requires the creation of an account to access. They have data available from 2007 to quarter 2, 2019. For the focus of our model we only want data about loans that have already either succesfully have been paid off or defaulted. Considering that the term of the loans are either 36 or 60 months, we will have to choose the data sets that are at least 5 years or older, so we will select data from 2007 to 2013. Let's begin by reading in the data.


Python code block:
```python
    import numpy as np

    def test_function(x,y):
      z = np.sum()
      return z
```

Here's some inline code `x+y`/

Here's some inline code `x+y`.

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/images/p2.jpg" alt="profile pic">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/images/images/p2.jpg)

Here's some math:

$$z=x+y$$

You can also put it inline $$z=x+y$$
