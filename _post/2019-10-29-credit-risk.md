---
title: "Data Science Project: Credit Risk Modeling"
date: 2019-01-28
tags: [data science, data analysis, machine learning, confusion matrix]
header:
  image:
excerpt: "Data Science, Credit Risk Modeling, Data Science"
---


{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Credit Risk Modelling Of Peer-to-Peer Lending Services\n",
    "\n",
    "Peer-to-peer lending is a newer form of lending which connects borrowers to lenders through various online marketplaces. This cuts out fees, protocols and other costs normally incurred by traditonal banks. Peer-to-peer lending is expected to grow very quickly, a report from [Transparency Market Research (TMR)](https://www.transparencymarketresearch.com/report-toc/10835) estimates a compound annual growth rate of 48.2% by 2024.\n",
    "\n",
    "## Focus\n",
    "\n",
    "We want to create a model that can predict whether a loan will be paid off on-time or not. The model needs to be built with simplicity with mind so it's workflow can quickly be adapted to growing and changing data.\n",
    "\n",
    "## Data\n",
    "\n",
    "We will use data from the peer-to-peer lending service, [Lending Club](https://www.lendingclub.com). This is ideal considering p2p lending is relatively new and we need data from at least a few years back which Lending Club has since it has been around for a while and they serve a lot customers.\n",
    "\n",
    "\n",
    "Each prospective borrower applies by providing financial history, the loans purpose, and other credit relevant data and Lending Club assigns an interest rate and grade to the loan or rejects the loan by using their own models. A loan with a higher interest rate with generate a higher return with a higher risk. Conversly, a loan with a lower interest will have lower risk and lower return. As conservative investors, we would like to create a model that can provide us with accurate and relevant predictions.\n",
    "\n",
    "\n",
    "## Understanding the Data\n",
    "\n",
    "Lending Club has loan data available on there [website](https://www.lendingclub.com/info/download-data.action), which requires the creation of an account to access. They have data available from 2007 to quarter 2, 2019. For the focus of our model we only want data about loans that have already either succesfully have been paid off or defaulted. Considering that the term of the loans are either 36 or 60 months, we will have to choose the data sets that are at least 5 years or older, so we will select data from 2007 to 2013. Let's begin by reading in the data.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# load in the data and libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline\n",
    "\n",
    "# data from 2007 - 2011\n",
    "loans_07 = pd.read_csv(\"LoanStats3a_securev1.csv\", low_memory=False, header=1)\n",
    "\n",
    "# data from 2012-2013\n",
    "loans_12 = pd.read_csv(\"LoanStats3b_securev1.csv\", low_memory=False, header=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(42538, 150)\n",
      "(188183, 150)\n"
     ]
    }
   ],
   "source": [
    "# display number of rows and columns of each data set\n",
    "print(loans_07.shape)\n",
    "print(loans_12.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Combining data\n",
    "\n",
    "Since these data sets share the same columns, we can merge them and clean the data as one as the data isn't too large."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(230721, 150)\n"
     ]
    }
   ],
   "source": [
    "# concat both data frames into a single data frame `loans`\n",
    "loans = pd.concat([loans_07, loans_12], axis=0).drop_duplicates().reset_index(drop=True)\n",
    "print(loans.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>member_id</th>\n",
       "      <th>loan_amnt</th>\n",
       "      <th>funded_amnt</th>\n",
       "      <th>funded_amnt_inv</th>\n",
       "      <th>term</th>\n",
       "      <th>int_rate</th>\n",
       "      <th>installment</th>\n",
       "      <th>grade</th>\n",
       "      <th>sub_grade</th>\n",
       "      <th>...</th>\n",
       "      <th>orig_projected_additional_accrued_interest</th>\n",
       "      <th>hardship_payoff_balance_amount</th>\n",
       "      <th>hardship_last_payment_amount</th>\n",
       "      <th>debt_settlement_flag</th>\n",
       "      <th>debt_settlement_flag_date</th>\n",
       "      <th>settlement_status</th>\n",
       "      <th>settlement_date</th>\n",
       "      <th>settlement_amount</th>\n",
       "      <th>settlement_percentage</th>\n",
       "      <th>settlement_term</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>1077501</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5000.0</td>\n",
       "      <td>5000.0</td>\n",
       "      <td>4975.0</td>\n",
       "      <td>36 months</td>\n",
       "      <td>10.65%</td>\n",
       "      <td>162.87</td>\n",
       "      <td>B</td>\n",
       "      <td>B2</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>N</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>1077430</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2500.0</td>\n",
       "      <td>2500.0</td>\n",
       "      <td>2500.0</td>\n",
       "      <td>60 months</td>\n",
       "      <td>15.27%</td>\n",
       "      <td>59.83</td>\n",
       "      <td>C</td>\n",
       "      <td>C4</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>N</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>1077175</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2400.0</td>\n",
       "      <td>2400.0</td>\n",
       "      <td>2400.0</td>\n",
       "      <td>36 months</td>\n",
       "      <td>15.96%</td>\n",
       "      <td>84.33</td>\n",
       "      <td>C</td>\n",
       "      <td>C5</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>N</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>1076863</td>\n",
       "      <td>NaN</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>36 months</td>\n",
       "      <td>13.49%</td>\n",
       "      <td>339.31</td>\n",
       "      <td>C</td>\n",
       "      <td>C1</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>N</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>1075358</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3000.0</td>\n",
       "      <td>3000.0</td>\n",
       "      <td>3000.0</td>\n",
       "      <td>60 months</td>\n",
       "      <td>12.69%</td>\n",
       "      <td>67.79</td>\n",
       "      <td>B</td>\n",
       "      <td>B5</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>N</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 150 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        id  member_id  loan_amnt  funded_amnt  funded_amnt_inv        term  \\\n",
       "0  1077501        NaN     5000.0       5000.0           4975.0   36 months   \n",
       "1  1077430        NaN     2500.0       2500.0           2500.0   60 months   \n",
       "2  1077175        NaN     2400.0       2400.0           2400.0   36 months   \n",
       "3  1076863        NaN    10000.0      10000.0          10000.0   36 months   \n",
       "4  1075358        NaN     3000.0       3000.0           3000.0   60 months   \n",
       "\n",
       "  int_rate  installment grade sub_grade  ...  \\\n",
       "0   10.65%       162.87     B        B2  ...   \n",
       "1   15.27%        59.83     C        C4  ...   \n",
       "2   15.96%        84.33     C        C5  ...   \n",
       "3   13.49%       339.31     C        C1  ...   \n",
       "4   12.69%        67.79     B        B5  ...   \n",
       "\n",
       "  orig_projected_additional_accrued_interest hardship_payoff_balance_amount  \\\n",
       "0                                        NaN                            NaN   \n",
       "1                                        NaN                            NaN   \n",
       "2                                        NaN                            NaN   \n",
       "3                                        NaN                            NaN   \n",
       "4                                        NaN                            NaN   \n",
       "\n",
       "  hardship_last_payment_amount  debt_settlement_flag  \\\n",
       "0                          NaN                     N   \n",
       "1                          NaN                     N   \n",
       "2                          NaN                     N   \n",
       "3                          NaN                     N   \n",
       "4                          NaN                     N   \n",
       "\n",
       "  debt_settlement_flag_date settlement_status settlement_date  \\\n",
       "0                       NaN               NaN             NaN   \n",
       "1                       NaN               NaN             NaN   \n",
       "2                       NaN               NaN             NaN   \n",
       "3                       NaN               NaN             NaN   \n",
       "4                       NaN               NaN             NaN   \n",
       "\n",
       "  settlement_amount settlement_percentage settlement_term  \n",
       "0               NaN                   NaN             NaN  \n",
       "1               NaN                   NaN             NaN  \n",
       "2               NaN                   NaN             NaN  \n",
       "3               NaN                   NaN             NaN  \n",
       "4               NaN                   NaN             NaN  \n",
       "\n",
       "[5 rows x 150 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# display the first 5 rows of the concatenated\n",
    "loans.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>member_id</th>\n",
       "      <th>loan_amnt</th>\n",
       "      <th>funded_amnt</th>\n",
       "      <th>funded_amnt_inv</th>\n",
       "      <th>term</th>\n",
       "      <th>int_rate</th>\n",
       "      <th>installment</th>\n",
       "      <th>grade</th>\n",
       "      <th>sub_grade</th>\n",
       "      <th>...</th>\n",
       "      <th>orig_projected_additional_accrued_interest</th>\n",
       "      <th>hardship_payoff_balance_amount</th>\n",
       "      <th>hardship_last_payment_amount</th>\n",
       "      <th>debt_settlement_flag</th>\n",
       "      <th>debt_settlement_flag_date</th>\n",
       "      <th>settlement_status</th>\n",
       "      <th>settlement_date</th>\n",
       "      <th>settlement_amount</th>\n",
       "      <th>settlement_percentage</th>\n",
       "      <th>settlement_term</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>230716</td>\n",
       "      <td>1059224</td>\n",
       "      <td>NaN</td>\n",
       "      <td>35000.0</td>\n",
       "      <td>35000.0</td>\n",
       "      <td>35000.0</td>\n",
       "      <td>36 months</td>\n",
       "      <td>15.96%</td>\n",
       "      <td>1229.81</td>\n",
       "      <td>C</td>\n",
       "      <td>C5</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>N</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>230717</td>\n",
       "      <td>1058722</td>\n",
       "      <td>NaN</td>\n",
       "      <td>12000.0</td>\n",
       "      <td>12000.0</td>\n",
       "      <td>12000.0</td>\n",
       "      <td>36 months</td>\n",
       "      <td>16.29%</td>\n",
       "      <td>423.61</td>\n",
       "      <td>D</td>\n",
       "      <td>D1</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>N</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>230718</td>\n",
       "      <td>1058291</td>\n",
       "      <td>NaN</td>\n",
       "      <td>12000.0</td>\n",
       "      <td>7775.0</td>\n",
       "      <td>7775.0</td>\n",
       "      <td>60 months</td>\n",
       "      <td>15.27%</td>\n",
       "      <td>186.08</td>\n",
       "      <td>C</td>\n",
       "      <td>C4</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>N</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>230719</td>\n",
       "      <td>Total amount funded in policy code 1: 2700702175</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>230720</td>\n",
       "      <td>Total amount funded in policy code 2: 81866225</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 150 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                      id  member_id  \\\n",
       "230716                                           1059224        NaN   \n",
       "230717                                           1058722        NaN   \n",
       "230718                                           1058291        NaN   \n",
       "230719  Total amount funded in policy code 1: 2700702175        NaN   \n",
       "230720    Total amount funded in policy code 2: 81866225        NaN   \n",
       "\n",
       "        loan_amnt  funded_amnt  funded_amnt_inv        term int_rate  \\\n",
       "230716    35000.0      35000.0          35000.0   36 months   15.96%   \n",
       "230717    12000.0      12000.0          12000.0   36 months   16.29%   \n",
       "230718    12000.0       7775.0           7775.0   60 months   15.27%   \n",
       "230719        NaN          NaN              NaN         NaN      NaN   \n",
       "230720        NaN          NaN              NaN         NaN      NaN   \n",
       "\n",
       "        installment grade sub_grade  ...  \\\n",
       "230716      1229.81     C        C5  ...   \n",
       "230717       423.61     D        D1  ...   \n",
       "230718       186.08     C        C4  ...   \n",
       "230719          NaN   NaN       NaN  ...   \n",
       "230720          NaN   NaN       NaN  ...   \n",
       "\n",
       "       orig_projected_additional_accrued_interest  \\\n",
       "230716                                        NaN   \n",
       "230717                                        NaN   \n",
       "230718                                        NaN   \n",
       "230719                                        NaN   \n",
       "230720                                        NaN   \n",
       "\n",
       "       hardship_payoff_balance_amount hardship_last_payment_amount  \\\n",
       "230716                            NaN                          NaN   \n",
       "230717                            NaN                          NaN   \n",
       "230718                            NaN                          NaN   \n",
       "230719                            NaN                          NaN   \n",
       "230720                            NaN                          NaN   \n",
       "\n",
       "        debt_settlement_flag debt_settlement_flag_date settlement_status  \\\n",
       "230716                     N                       NaN               NaN   \n",
       "230717                     N                       NaN               NaN   \n",
       "230718                     N                       NaN               NaN   \n",
       "230719                   NaN                       NaN               NaN   \n",
       "230720                   NaN                       NaN               NaN   \n",
       "\n",
       "       settlement_date settlement_amount settlement_percentage settlement_term  \n",
       "230716             NaN               NaN                   NaN             NaN  \n",
       "230717             NaN               NaN                   NaN             NaN  \n",
       "230718             NaN               NaN                   NaN             NaN  \n",
       "230719             NaN               NaN                   NaN             NaN  \n",
       "230720             NaN               NaN                   NaN             NaN  \n",
       "\n",
       "[5 rows x 150 columns]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# display the last 5 rows\n",
    "loans.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "id                        object\n",
       "member_id                float64\n",
       "loan_amnt                float64\n",
       "funded_amnt              float64\n",
       "funded_amnt_inv          float64\n",
       "                          ...   \n",
       "settlement_status         object\n",
       "settlement_date           object\n",
       "settlement_amount        float64\n",
       "settlement_percentage    float64\n",
       "settlement_term          float64\n",
       "Length: 150, dtype: object"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# observe data types of the features\n",
    "loans.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Target column:\n",
    "\n",
    "Using the [data dictionary](https://resources.lendingclub.com/LCDataDictionary.xlsx), we can define our target column as `loan_status` because it describes whether a loan was paid off on time, was delayed, or went into default. This column contains text values which we will need to convert to numeric ones."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fully Paid                                             192626\n",
      "Charged Off                                             35341\n",
      "Does not meet the credit policy. Status:Fully Paid       1988\n",
      "Does not meet the credit policy. Status:Charged Off       761\n",
      "Name: loan_status, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# display the categorical values\n",
    "print(loans['loan_status'].value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Only the `Fully Paid` and `Charged Off` values describe the outcome fo the loan. The other categories describe loans that are still ongoing and thus will no add any value to our models or are in some new payment agreement, regardless there just aren't enough of these cases to warrant keeping them for our model. By using just the `Fully Paid` and `Charged Off` values we can turn this into a **binary classification** problem. We need to transform `Fully Paid`'s value into 1 for a positive case and `Charge Off` into 0 to represent a negative case. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# select only 'Fully Paid' and 'Charged Off' data\n",
    "loans = loans[(loans['loan_status'] == 'Fully Paid') | (loans['loan_status'] == 'Charged Off')]\n",
    "# create replacement dictionary to convert to numerical\n",
    "status_replace = {\n",
    "    'loan_status' : {\n",
    "        'Fully Paid' : 1,\n",
    "        'Charged Off' : 0,\n",
    "    }\n",
    "}\n",
    "# replace values to numerical values\n",
    "loans = loans.replace(status_replace)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1    192626\n",
      "0     35341\n",
      "Name: loan_status, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# display the new target column values\n",
    "print(loans['loan_status'].value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Cleaning\n",
    "\n",
    "The data has 230721 rows and a 150 columns with varying amounts of data types, this will need to be reduced somehow. The main focus of cleaning is for performance in our machine learning models. The [data dictionary](https://resources.lendingclub.com/LCDataDictionary.xlsx) can give us some insight on which columns to remove:\n",
    "\n",
    "- non-contributing columns: columns where the information does not alter the ability of the borrower to pay back the loan like ID values for example\n",
    "- data format: data formatted poorly will need to be cleaned up or some data may be too time intensive\n",
    "- data redundancy: any data that repeats information found possibly in another column\n",
    "- data leakage: data from the future that could disrupt our model, columns with data containing information about the loan after it has been funded like whether it has been paid off or not.\n",
    "\n",
    "We will use data visualization for further analysis on whether a column will work in our model or not. After this we can again use data visualization to help remove any missing vales, create categorical data for columns, and find any correlations between our feature and target columns.\n",
    "\n",
    "### Non-contributing columns & data format:\n",
    "\n",
    "We will eliminate any non-contributing columns. Things like `id`, `member_id`, `url`, and `policy_code` will not add any value to our models and are randomely generated by Lending Club. We will also remove columns that may have data like `emp_title`, `desc`, `debt_settlement_flag`, `issue_d`, or `initial_list_status` that would require complex analysis to make useful and still would not likely benefit our models."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove non-contributing columns and columns with unacceptable data formats\n",
    "loans = loans.drop(['id', 'member_id', 'emp_title', 'url', 'policy_code', 'desc', 'debt_settlement_flag', \n",
    "                    'issue_d', 'initial_list_status'], axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data redundancy:\n",
    "\n",
    "The next step is to sort through redundant data. The `zip_code` column only contains 3 digits to protect the privacy of the borrowers, these 3 digits only give the state in which the borrower resides in making this column contain the same information as `addr_state`. Columns `grade` and `sub-grade` have data contained in `int_rate` so we will need to decide which one to use. We can graph the correlations differences for some insight. We will need to convert `int_rate` to numeric data and strip it's `%` sign."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# convert columns to numeric data\n",
    "loans['int_rate'] = loans['int_rate'].str.rstrip('%').astype('float')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's look at the `int_rate` of each grade."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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se6j+PNyRVIiIiIiIvqLY1VFREREREREDoLFHhERERERUQXkEMXexo3rMGvWdJv2mZ+fj88+G3vfa+LidmPDhl9tE4iIiIiIiByKQxR7csjLy8WlSxfve8358+eg1RbZKBERERERETmSCrf1wv1s27YFCxfOhxACHTs+hf79X0Zubi4mT/4UN27cQE5ODoYOfROPP94Rb745GLVrR+PIkUNwclJjwoTJCAgIuGu758+fw9Spn0Or1SIyMgojR47BnDkzkZx8CdOnT8HAgUPu6KNOnWisXbsKSqUSVapUwxdffIoff1wKNzc3zJ//LXx8fPDkk53xyScfITs7G+7u7hg9egKCgoJtfNeIiIiIiKg8cpiRvby8XHz33beYPXse5s//CfHxcTh+/Cji4/eiWbOWmDdvIaZMmY4ffphX8h4fHx/Mn78IdevWx8aN955uuWLFEgwe/AYWLVqGoKBgpKamYOjQt1CtWg28884Hd+0jJCQU3bv3RL9+A9C0abO7trtnz07Urh2NH374Gd27P4ezZ09b/L4QEREREVHF5DAjewkJF9CwYWN4eXkDADp06IijRw/jpZdexZEjh7B48UKcO3cWWu1f+9U1bhwLAKhWrTqSk5Pu2XbTps3w6aej0bZtBzzxxJOoUqUa0tJSS84/9VTne/ZxP3Xq1MW3387CtWtX0bbt42jTpv3DfHQiIiIiInJADjOyV1CQX+q1JEkwmUz4+edFWLp0MSpXroKXX34VkvTXfnVOTmoAxZuX//34P3Xq9DS++WY+QkLC8Pnn47B3765S5+/Xx23FG8AXHzcajQCKi8xFi5ajWbOWWLVqGWbPtu0iM0REREREVH45TLFXv35DHDnyB3Jzb8FgMGDnzm2oV68+jh07gueffwFt27bHsWNHYTaby9z26NEf4+bNDLzwQj88/XQXXLp0EUqlEiaTCQDu2cffr/Hy8kZychKMRiMOHToAAFi5chlWr16OZ57pisGDX8fFi4kWuhtERERERFTROcw0Ti8vb7zyyhC8+eZgGI1GdOjQEbGxzQEAU6dOglqtQs2aGigUCuh0DzbN8rb//rc/Jk/+DIBAQEAARo0aBzc3d6jVKnz++Tj06vWfu/YRE9MAEydOQEhIKAYMeAVjxoxEQEAAatbUAAA6dnwKn3wyAjt3boezszOGD//I0reFiIiIiIgqKCHd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R3TzzTdr+fLl+vWvf62qqio9+uijkqQjR47o2muvVVJSkhYtWqTbb79dCxcuPO38TzzxhIqKirRgwQItXrxYF198sb73ve9pzZo13ffCAQB9EtM4AQB9Rl1dnSQpPj7+tPvnzJmjysrK9s9nz56twsJCSdLNN9+s/v37S5Lq6+v18MMPa+bMmZKk9PR0XXDBBVqyZIkk6c0331Rra6seeughRUZGauDAgaqtrdWCBQskSU1NTXrxxRf19NNPa8qUKZLaymVpaamef/55jRs3rhtfPQCgr6HsAQD6jLi4OEltI3CftXDhQrW2tkqS7rzzTrW0tEiSDMNQRkZG++POOuss7dixQ88884zKysq0e/du7dixQ6mpqZKknTt3Kjs7W5GRke3HnCyNkrRr1y61tLTotttuk8NxanJNa2urkpKSuvjVAgD6OsoeAKDP8Hq9SkpK0rp169pH5yTJ4/G03w4LC2u/7XA45Ha72z9fvHix7r77bs2ZM0djxozRVVddpZUrV7aP7BmGIcuyTnvOkJCQ9tsnz/X0008rMzPztMd9tvwBANAV+MkCAOgznE6n5s6dq6KiIu3atevf/r6lpUUHDx78wuNffPFFXXHFFXr44Yd15ZVXatSoUaqoqGgveEOGDFFZWdlpI4dbt25tv52ZmamQkBDV1tYqMzOz/WPp0qUqKirqwlcKAABlDwDQx3znO9/RhAkT9K1vfUsvvPCCdu7cqcrKSi1dulSXXHKJysrKNHr06M89NiEhQevXr1dpaanKy8v1zDPPaNmyZe3TPmfMmKHY2Fjdcccd2rFjh1atWqWnnnqq/fjw8HBde+21euKJJ7Rs2TJVVlbqxRdf1LPPPtu+LhAAgK5iWP863wQAgCBnWZYWL16soqIibd++XceOHZPH49HkyZN19dVXKysrS0VFRZo/f75KSkraj6uoqND8+fO1efNmhYeHa8SIEfr617+ue++9V++++648Ho/27NmjBQsWaN26dUpJSdG1116rBQsW6J133lFGRoZ8Pp+effZZvf7666qrq1P//v11/fXX69JLL7XxXwQAEIwoewAAAAAQhJjGCQAAAABBiLIHAAAAAEGIsgcAAAAAQYiyBwAAAABBiLIHAAAAAEGIsgcAAAAAQYiyBwAAAABBiLIHAAAAAEGIsgcAAAAAQej/A3YSu2EU2dKFAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 1080x720 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# graph similarites of `int_rate` and `grade` columns\n",
    "sns.set(rc={'figure.figsize':(15,10)})\n",
    "sns.violinplot(x=\"grade\", y=\"int_rate\", data=loans, palette='viridis', order=\"ABCDEFG\",hue='loan_status',split=True)\n",
    "plt.title(\"Interest Rate - Grade\", fontsize=20)\n",
    "plt.xlabel(\"Grade\", fontsize=15)\n",
    "plt.ylabel(\"Interest Rate\", fontsize=15);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We see that `grade`, for the most part, categorizes the `int_rate` column. We do see outliers and uneven distribution which indicates that Lending Club's grade of the loan also includes other data. The distributions are mostly evenly distributed with respect to lower grade and higher interest rate but non-evenly distributed in higher grades and lower interest rates which also indicates the complexity of the `grade` column. To make our model simpler, we will remove `grade` and keep `int_rate`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove columns containing data redundancy\n",
    "loans = loans.drop(['zip_code', 'grade', 'sub_grade'], axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data leakage:\n",
    "\n",
    "We need to remove any columns that contain any information that could leak data to the prediction model. So columns: `funded_amnt`, `funded_amnt_inv`, `out_prncp`, `out_prncp_inv`, `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `recoveries`, `collection_recovery_fee`, `last_pymnt_d`, and `last_pymnt_amnt` all give away information on whether the loan was succesfully paid off or not which would disrupt our model from correctly predicting that information correctly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove columns that leak data from the future and disrupt our prediction models\n",
    "loans = loans.drop([\n",
    "    'funded_amnt', 'funded_amnt_inv', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', \n",
    "    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', \n",
    "    'last_pymnt_d', 'last_pymnt_amnt'\n",
    "                         ], axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Unique values:\n",
    "\n",
    "Let's remove any columns that contain only one unique value as they won't be useful for any of our machine learning models. Columns with only one unique value will not help our model make predictions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['pymnt_plan', 'application_type', 'hardship_flag', 'hardship_type', 'deferral_term', 'hardship_length']\n"
     ]
    }
   ],
   "source": [
    "# remove any columns that only contain one unique value and display\n",
    "orig_columns = loans.columns\n",
    "drop_columns = []\n",
    "for col in orig_columns:\n",
    "    col_series = loans[col].dropna().unique()\n",
    "    if len(col_series) == 1:\n",
    "        drop_columns.append(col)\n",
    "        \n",
    "loans = loans.drop(drop_columns, axis=1)\n",
    "print(drop_columns)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Feature Preperation\n",
    "\n",
    "Our machine learning models will require the columns in the data to be numeric. Columns with missing values will also not work. We will go through our data and prepare so that it can succesfully go through our model.\n",
    "\n",
    "### Null values:\n",
    "\n",
    "Let's see which columns have null values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "emp_length                  8965\n",
      "title                         18\n",
      "mths_since_last_delinq    133300\n",
      "mths_since_last_record    207702\n",
      "revol_util                   175\n",
      "                           ...  \n",
      "settlement_status         225739\n",
      "settlement_date           225739\n",
      "settlement_amount         225739\n",
      "settlement_percentage     225739\n",
      "settlement_term           225739\n",
      "Length: 95, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# display columns with null values\n",
    "null_counts = loans.isnull().sum()\n",
    "print(null_counts[null_counts>0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A lot of columns are missing data. Let's remove any column that is missing more than 50% of its data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "emp_length                     8965\n",
      "title                            18\n",
      "revol_util                      175\n",
      "last_credit_pull_d               12\n",
      "collections_12_mths_ex_med       56\n",
      "tot_coll_amt                  67527\n",
      "tot_cur_bal                   67527\n",
      "total_rev_hi_lim              67527\n",
      "acc_open_past_24mths          47281\n",
      "avg_cur_bal                   67533\n",
      "bc_open_to_buy                48811\n",
      "bc_util                       48898\n",
      "chargeoff_within_12_mths         56\n",
      "mo_sin_old_il_acct            73658\n",
      "mo_sin_old_rev_tl_op          67528\n",
      "mo_sin_rcnt_rev_tl_op         67528\n",
      "mo_sin_rcnt_tl                67527\n",
      "mort_acc                      47281\n",
      "mths_since_recent_bc          48614\n",
      "mths_since_recent_inq         67654\n",
      "num_accts_ever_120_pd         67527\n",
      "num_actv_bc_tl                67527\n",
      "num_actv_rev_tl               67527\n",
      "num_bc_sats                   55841\n",
      "num_bc_tl                     67527\n",
      "num_il_tl                     67527\n",
      "num_op_rev_tl                 67527\n",
      "num_rev_accts                 67527\n",
      "num_rev_tl_bal_gt_0           67527\n",
      "num_sats                      55841\n",
      "num_tl_120dpd_2m              67783\n",
      "num_tl_30dpd                  67527\n",
      "num_tl_90g_dpd_24m            67527\n",
      "num_tl_op_past_12m            67527\n",
      "pct_tl_nvr_dlq                67680\n",
      "percent_bc_gt_75              48814\n",
      "pub_rec_bankruptcies            697\n",
      "tax_liens                        39\n",
      "tot_hi_cred_lim               67527\n",
      "total_bal_ex_mort             47281\n",
      "total_bc_limit                47281\n",
      "total_il_high_credit_limit    67527\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# remove any columns that contain 50% or more NaN values\n",
    "pct_null = loans.isnull().sum() / len(loans)\n",
    "missing_features = pct_null[pct_null > 0.50].index\n",
    "loans.drop(missing_features, axis=1, inplace=True)\n",
    "\n",
    "# display columns with remaining NaN values\n",
    "null_counts = loans.isnull().sum()\n",
    "print(null_counts[null_counts>0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There are still many columns with exactly the same amount of missing data left. For the purposes of our model and the desire to expand the model quickly in the future, it is best to remove any column with just 15% or more missing values. Columns with large amounts of missing data will presumambly also be left blank in future loans made by Lending Club or similiar services."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "emp_length                    8965\n",
      "title                           18\n",
      "revol_util                     175\n",
      "last_credit_pull_d              12\n",
      "collections_12_mths_ex_med      56\n",
      "chargeoff_within_12_mths        56\n",
      "pub_rec_bankruptcies           697\n",
      "tax_liens                       39\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# remove any columns that contain just 15% or more NaN values\n",
    "pct_null = loans.isnull().sum() / len(loans)\n",
    "missing_features = pct_null[pct_null > 0.15].index\n",
    "loans.drop(missing_features, axis=1, inplace=True)\n",
    "\n",
    "# display columns with remaining NaN values\n",
    "null_counts = loans.isnull().sum()\n",
    "print(null_counts[null_counts>0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The `emp_length` and `pub_rec_bankruptcies` variables still have a lot of null values, let's explore the data for these columns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.0    0.921445\n",
      "1.0    0.074081\n",
      "NaN    0.003057\n",
      "2.0    0.001127\n",
      "3.0    0.000167\n",
      "4.0    0.000079\n",
      "5.0    0.000018\n",
      "6.0    0.000018\n",
      "7.0    0.000004\n",
      "8.0    0.000004\n",
      "Name: pub_rec_bankruptcies, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "# display the number of unique values in `pub_rec_bankruptcies` column\n",
    "print(loans.pub_rec_bankruptcies.value_counts(normalize=True, dropna=False))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Considering that around 92% of the values are unique, this column will not add any value to our predective models so we will drop this column. The `emp_length` is too important to our models to drop, so we will just remove the missing data from this column, along with any of the other missing data from the other columns since they aren't missing much data compared to the other columns we have removed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "# drop `pub_rec_bankruptcies` and any remaining Nan values in the remaining columns of the data set\n",
    "loans = loans.drop('pub_rec_bankruptcies', axis=1)\n",
    "loans = loans.dropna(axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "loan_amnt                     0\n",
       "term                          0\n",
       "int_rate                      0\n",
       "installment                   0\n",
       "emp_length                    0\n",
       "home_ownership                0\n",
       "annual_inc                    0\n",
       "verification_status           0\n",
       "loan_status                   0\n",
       "purpose                       0\n",
       "title                         0\n",
       "addr_state                    0\n",
       "dti                           0\n",
       "delinq_2yrs                   0\n",
       "earliest_cr_line              0\n",
       "fico_range_low                0\n",
       "fico_range_high               0\n",
       "inq_last_6mths                0\n",
       "open_acc                      0\n",
       "pub_rec                       0\n",
       "revol_bal                     0\n",
       "revol_util                    0\n",
       "total_acc                     0\n",
       "last_credit_pull_d            0\n",
       "last_fico_range_high          0\n",
       "last_fico_range_low           0\n",
       "collections_12_mths_ex_med    0\n",
       "acc_now_delinq                0\n",
       "chargeoff_within_12_mths      0\n",
       "delinq_amnt                   0\n",
       "tax_liens                     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# display columns with remaining NaN values\n",
    "loans.isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We have removed or filled all missing values.\n",
    "\n",
    "### Converting to numerical data:\n",
    "\n",
    "We will now convert our object data into numeric values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "float64    20\n",
      "object     10\n",
      "int64       1\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# display the different types of data in our remaing columns\n",
    "print(loans.dtypes.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "term                     36 months\n",
      "emp_length               10+ years\n",
      "home_ownership                RENT\n",
      "verification_status       Verified\n",
      "purpose                credit_card\n",
      "title                     Computer\n",
      "addr_state                      AZ\n",
      "earliest_cr_line          Jan-1985\n",
      "revol_util                   83.7%\n",
      "last_credit_pull_d        May-2019\n",
      "Name: 0, dtype: object\n"
     ]
    }
   ],
   "source": [
    "# display the data that are `object` types\n",
    "object_columns_df = loans.select_dtypes(include=['object'])\n",
    "print(object_columns_df.iloc[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "According the [data dictionary](https://resources.lendingclub.com/LCDataDictionary.xlsx), a lot of these values can be translated into categorical values which we can run through our model:\n",
    "\n",
    "- home_ownership: home ownership status, can only be 1 of 4 categorical values according to the data dictionary\n",
    "- verification_status: indicates if income was verified by Lending Club\n",
    "- emp_length: number of years the borrower was employed upon time of application\n",
    "- term: number of payments on the loan, either 36 or 60\n",
    "- addr_state: borrower's state of residence\n",
    "- purpose: a category provided by the borrower for the loan request\n",
    "- title: loan title provided the borrower\n",
    "\n",
    "Let's verify through checking the amount of unique values in each of these columns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MORTGAGE    110366\n",
      "RENT         91241\n",
      "OWN          16961\n",
      "OTHER          139\n",
      "NONE            42\n",
      "Name: home_ownership, dtype: int64\n",
      "Verified           93725\n",
      "Not Verified       74887\n",
      "Source Verified    50137\n",
      "Name: verification_status, dtype: int64\n",
      "10+ years    70046\n",
      "2 years      20518\n",
      "3 years      18049\n",
      "< 1 year     17767\n",
      "5 years      17538\n",
      "1 year       14653\n",
      "4 years      14269\n",
      "6 years      14135\n",
      "7 years      12969\n",
      "8 years      10429\n",
      "9 years       8376\n",
      "Name: emp_length, dtype: int64\n",
      " 36 months    165239\n",
      " 60 months     53510\n",
      "Name: term, dtype: int64\n",
      "CA    36458\n",
      "NY    19237\n",
      "TX    16748\n",
      "FL    14938\n",
      "NJ     8786\n",
      "IL     8520\n",
      "PA     7574\n",
      "VA     6973\n",
      "GA     6971\n",
      "OH     6791\n",
      "NC     5911\n",
      "MA     5579\n",
      "MD     5158\n",
      "WA     5123\n",
      "MI     4946\n",
      "AZ     4863\n",
      "CO     4576\n",
      "MN     3716\n",
      "CT     3531\n",
      "MO     3481\n",
      "NV     3091\n",
      "OR     2864\n",
      "AL     2648\n",
      "WI     2642\n",
      "LA     2595\n",
      "SC     2471\n",
      "IN     2212\n",
      "KS     1984\n",
      "TN     1948\n",
      "KY     1935\n",
      "OK     1889\n",
      "UT     1696\n",
      "AR     1579\n",
      "HI     1240\n",
      "NM     1130\n",
      "NH     1027\n",
      "WV     1013\n",
      "RI      971\n",
      "DC      763\n",
      "MT      627\n",
      "AK      625\n",
      "DE      558\n",
      "WY      521\n",
      "SD      455\n",
      "VT      343\n",
      "MS       22\n",
      "ID        7\n",
      "IA        6\n",
      "NE        5\n",
      "ME        2\n",
      "Name: addr_state, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# display the unique values of the categorical columns\n",
    "unique_ct = ['home_ownership', 'verification_status', 'emp_length', 'term', 'addr_state']\n",
    "\n",
    "for c in unique_ct:\n",
    "    print(loans[c].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Debt consolidation         22376\n",
      "Debt Consolidation         15055\n",
      "Credit card refinancing     7451\n",
      "Consolidation               5096\n",
      "debt consolidation          4430\n",
      "                           ...  \n",
      "High Hopes                     1\n",
      "Lot Development                1\n",
      "All My Loans                   1\n",
      "Help Me Get Married            1\n",
      "My DIY Wedding                 1\n",
      "Name: title, Length: 58491, dtype: int64\n",
      "debt_consolidation    125492\n",
      "credit_card            46097\n",
      "home_improvement       12592\n",
      "other                  12149\n",
      "major_purchase          5594\n",
      "small_business          4465\n",
      "car                     3356\n",
      "wedding                 2244\n",
      "medical                 2094\n",
      "moving                  1539\n",
      "house                   1410\n",
      "vacation                1199\n",
      "educational              309\n",
      "renewable_energy         209\n",
      "Name: purpose, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# display the data in `title` and `purpose` columns\n",
    "print(loans[\"title\"].value_counts())\n",
    "print(loans[\"purpose\"].value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The `last_credit_pull_d`, `addr_state`, `earliest_cr_line`, and `title` columns will not add much predictive power to our models or are overly complex so we will remove these columns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove columns that will not contribute to the predictive models\n",
    "loans = loans.drop(['last_credit_pull_d', 'addr_state', 'earliest_cr_line', 'title'], axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The `revol_util` column needs to be converted from an interest rate to a float."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "# convert column to numeric data\n",
    "loans['revol_util'] = loans['revol_util'].str.rstrip('%').astype('float')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The `emp_length`, `home_ownership`, `verification_status`, and `term` can be converted into categorical values which will be more effective in relation to our predictive power. We encode these columns as dummy variables in order to do this.\n",
    "\n",
    "Let's graph the distrubtion of `emp_length` for issued loans to help categorize this variable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA4kAAAHwCAYAAAALnCp1AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAgAElEQVR4nOzde5zWdYH3//fADIM2FIEzgKTmwzRLTKrpQOagbbeAgibJfbuQh04eblfNWmwE0lCJNBZQE8taO2itkYdBaRy37EZLzEXvbdVN1HVREbhhOCgHBeb0+8OH3x+jiKBejMrz+Xj4GK7PdX2v+Xy+M1O8+B6mrKOjoyMAAACQpFtXTwAAAIC3DpEIAABAQSQCAABQEIkAAAAURCIAAAAFkQgAAEBBJAK8Ac8880w+9KEP5dhjj82xxx6bUaNG5YQTTkhjY2PxmssvvzwNDQ3bfJ8f/vCH+cMf/rDV57bc/oMf/GBWr169Q3N88MEHc8EFFyRJHnrooZx99tk7tP3r0dbWljPOOCPDhg3L9ddf3+m5m2++OR//+MeLffbSf+edd96b9vlfz37qalt+nV7uxBNPTFNT006Zx1e+8pVi333uc5/LQw899Jrb1NfX57DDDnvF13T58uWvex733XdfRo4c+bq3fzN89KMfzTPPPPOK8Z359QDoCuVdPQGAt7uePXtmzpw5xeMlS5bklFNOSffu3TNs2LCcc845r/ke9913Xz7wgQ9s9bnt2X5b/uu//qv4y/rBBx+cK6644g293/ZYvnx5/vznP+evf/1runfv/orna2tr8+Mf/7jk83g72fLr1JXuueee17XdKaeckq9+9atv8mwA6AoiEeBNNnDgwJx99tn553/+5wwbNiz19fXZf//989WvfjVXXHFFfv/736eioiLvfe97M3Xq1Pz+97/Pww8/nMsuuyzdu3fPnXfemWeffTaLFy/O4YcfnlWrVhXbJ8nMmTPz0EMPpb29Pd/4xjdyxBFH5Oabb84dd9xRhNdLj7/73e/miiuuyLp163L++efnC1/4Qi6++OLMnTs369aty+TJk7Nw4cKUlZXlsMMOyze/+c2Ul5fn4IMPzqmnnpp77rknK1asyNe+9rWMHTv2FWu9//77c9lll+WFF15IRUVFvvGNb+RjH/tYvva1r6W1tTWjR4/OlVdemb333nu79199fX169uyZxx57LKtWrcrnPve59O7dO//n//yfNDc355JLLsmQIUNSX1+fysrKLFy4MKtWrcqhhx6aSZMmpaKiotP7XXXVVfnd736X7t27Z9999813vvOdtLS0ZOTIkbnrrrvSq1evdHR0ZPjw4bn88sszcODATJkyJY899lhaWloyZMiQnHfeecV++fKXv5z58+fn+eefzz/8wz+kqakpjz32WGpqavKjH/0ou+++e5544olMmTIlzz77bNra2nLiiSfm+OOPz3333ZcZM2Zkr732yuOPP57W1tZMnjw5e+65Z6ev09SpU7d7f/3xj3/M1VdfnZaWlvTs2TPf/va389GPfjRXXnlllixZkubm5ixZsiT9+vXLD37wg9TU1OTBBx/Md7/73bS0tGTvvffO0qVLU19fXxyxPvnkk3PNNdckSX7zm9/kwgsvzOrVq3Psscfm3HPP3e65Jdnm99mgQYPyd3/3d1m4cGGmTZuWgw8+eKvv0dzcnG9/+9tZs2ZNkmTo0KH5xje+8arjr/bz8OMf/zibN2/OtGnTsmDBgrS1teXDH/5wJk2alKqqqtx///25+OKLU1ZWloMPPjjt7e07tNaX9td1112Xbt26ZY899sh3vvOd7Lvvvlm0aFEuuuiibNiwIc3NzTnwwAMzc+bMVFZWvurP26utD6DUnG4KUAIHHnhgHnvssU5jy5Ytyy9+8YvcdNNNufnmm3PooYfmwQcfzLhx4zJo0KCcd955+R//438kSTZu3Jjf/e53GT9+/Cve+33ve19uueWW/OAHP0h9ff02T6scMGBAzj777NTW1r4iPC655JL07t07t912W2666aY8+uijufbaa5Mkmzdvznvf+97ccMMNueKKKzJ16tRs2rSp0/Zr1qzJ2WefnYkTJ+a2227LpZdemvHjx2fNmjW55ppriiOsWwvE+++//xWnJt50003F83/729/yi1/8Itdff32uvfba7L777rnhhhty0kkn5Sc/+UnxugcffDDXXnttGhsb88QTT+Q3v/lNp89z00035U9/+lNuvPHG3Hbbbdl///1TX1+fPffcM5/+9Kdz6623Jkn+8pe/pHfv3jnwwAPzve99LwcddFBuvvnmNDQ0ZM2aNfnZz35W7Jc99tgjN954Y77whS9k0qRJmThxYhobG7N+/frceeedaW1tzdlnn51vfetbufnmm4s1/PWvfy3m/JWvfCUNDQ0ZPXp0ZsyYsc2v07Y8+eSTmTFjRq655po0NDTk4osvzllnnZXnn3++2M+XX355mpqasttuu+WGG25Ia2trzjrrrJxzzjm57bbbcuKJJ+aRRx5JkuJz/+IXv8iAAQOSJJWVlbn55pvz29/+Ntdee22WLVu21bn8/Oc/7/T1/O1vf5tk299nLS0tOeKII3LHHXe8aiAmyezZs4vv+1/96ld56qmnsm7dulcd35Zrrrkm3bt3z80335xbb701NTU1mTZtWjZv3pxzzjmniOVPfepT2bhx43Z/LZLk3nvvzU9/+tP88pe/zK233pqRI0fmzDPPTEdHR2bPnp0vfOELmT17dv71X/81zzzzTObNm5fk1X/eXs/6AN4MjiQClEBZWVl69uzZaaxfv3458MADc9xxx6Wuri51dXUZMmTIVrf/+Mc//qrv/fd///dJkgMOOCD77bdf/v3f//11zfHuu+/Ov/zLv6SsrCw9evTICSeckF/84hc59dRTkyR/93d/lyQ56KCDsnnz5jz//POprKwstn/wwQez995755BDDkmS7L///vnYxz6Wf/u3f8unPvWpbX7u1zrd9IgjjkhFRUWqq6uz++6757DDDkuS7L333nn22WeL1x133HF517velSQ59thjc+edd+ZLX/pSpzWOHj06u+++e5LkpJNOyo9+9KNs3rw548aNyw9+8IOMGzcuv/nNb4r9Om/evDz00EO58cYbk+QVoTBs2LBiLgcccED69euX5MV4f+655/Lkk0/m6aefzoQJE4ptNm7cmL/97W/Zb7/9sueee+ZDH/pQkuTDH/5wbrnllm3uq2156cjTKaecUoyVlZXl6aefTpJ88pOfTFVVVfG5nnvuueIfL4YOHZok+fSnP53999//VT/HS9cFVldXZ4899siqVauKgNzSq51u+lrfZ7W1ta+5zsMOOyynnnpqli1bls985jP51re+lV69er3q+LbMmzcv69aty/z585O8GKp9+/bNY489lvLy8uJncuTIka96jeir+dOf/pSjjjoqffr0SZKMHj06U6ZMyTPPPJPx48fnnnvuyU9+8pM8+eSTWbFiRRHzydZ/3l7P+gDeDCIRoAQeeuihHHDAAZ3GunXrluuvvz4PPfRQ7r333nzve9/LYYcdttUbtrwUNVvTrdv/fxJIe3t7ysvLU1ZWlo6OjmK8paXlNefY3t6esrKyTo9bW1uLxy8F4Uuv2fL9kxdvTrPl9i+9Zsv3eL169OjR6XF5+db/72rL6x07Ojo67Ztk22v8zGc+kxdeeCH33ntv7r///lx66aXFay6//PLst99+SZK1a9d2eo8tT2d9+amtyYv7pVevXp2uU125cmV69eqVv/71r53+8eDlX7cd1d7eniFDhmTmzJnF2LJly1JTU5Pf//73W/1c3bt3f8Xn3Np1oy/Zct+/nvm+1vfZtr7XX/KRj3wkd955Z+6999785S9/yZgxY/KTn/zkVce39fPQ3t6eCRMmFJG8YcOGbNq0KUuXLn3F2l7t+25ba325l34mvvnNb6atrS0jRozI4YcfnmXLlnX6fFv7eXu19Q0aNGiH5gWwo5xuCvAmW7RoUWbNmpWvfOUrncYXLlyYkSNHZr/99stpp52WU045pbhzZPfu3bc7rl468vSf//mfefrpp3PIIYekT58+efzxx7Np06a0tLTkjjvuKF7/au/92c9+Ntdff306OjqyefPmzJ49O5/5zGe2e52DBw/Of//3f+fBBx9Mkjz++ONZsGBBPvnJT273e7xRt99+ezZv3pxNmzbllltuyRFHHNHp+cMOOyw33XRTccTmuuuuyyc+8Yn06NEjZWVlGTt2bCZOnJiRI0cWf0n/7Gc/m5///OfFfjnjjDNecYfWbdl333073cxo2bJlGTlyZB5++OFtbrcj3wMvGTJkSO6555488cQTSZK77rorxxxzzDZPk9xvv/3So0eP3H333UlePCL82GOPFXHyeuaxLW/0+yxJpk2bllmzZuXzn/98Jk6cmA984AN5/PHHX3V8Wz8Pn/3sZ/OrX/0qmzdvTnt7e77zne9k+vTp+eAHP5iOjo7cddddSZI777wzzz333A7N87DDDktjY2NxCvhNN92U3r17Z5999smf//znnHnmmTnqqKOSJP/xH/+Rtra217VugFJzJBHgDdq4cWOOPfbYJC8e5ausrMw3v/nNHH744Z1ed+CBB2bEiBH54he/mN133z09e/bMpEmTkrz4qwamT5++XUcAFy9enC984QspKyvL9OnT07t37xx66KH5xCc+kREjRqS6ujqf+tSn8uijjyZ5Meauuuqq/MM//ENOPPHE4n0mTZqUSy65JKNGjUpLS0sOO+ywnH766du97j59+uTyyy/PxRdfnI0bN6asrCxTp07Nvvvuu9VfG7Cll65J3NJL14ntiJ49e2bs2LFZu3Zthg0bli9+8Yudnj/++OOzbNmyjBkzJu3t7dlnn30ybdq04vnjjjsul156af7X//pfxdjEiRMzZcqUYr985jOfyde+9rXtnlOPHj0ya9asTJkyJT/96U/T2tqac845Jx//+Mdz3333vep2W36dfvjDH77i+fPOOy/nn39+8Xjs2LEZP358Lrroonzzm99MR0dHysvLc/XVVxen4G5NeXl5rrzyylx44YWZPn163v/+92ePPfYojjoOHz48J554Yq688srtXvO2vNHvs+TFG+nU19dn5MiR6dGjRz74wQ/m6KOPznPPPbfV8W7dur3qz8P//t//O5deemmOO+64tLW15UMf+lDq6+tTUVGRq666Kt/97nczffr0fOhDH0rfvn1fdU6v9vU45ZRTcvLJJ6e9vT19+vTJj3/843Tr1i3nnntuzjzzzOy+++6pqqrKJz7xieK04B1dN0CplXW8kfNcAKCLbHnX2Nfrd7/7XW655Zb89Kc/fRNn9tZ36aWX5qtf/Wr22GOPLFu2LMcee2z+8Ic/5N3vfndXTw2AtwBHEgHYJZ144olZvXp1Zs2a1dVT2ekGDhyYU045JeXl5eno6Mgll1wiEAEoOJIIAABAwY1rAAAAKIhEAAAACiIRAACAwi5745o1azakvd3lmAAAwK6lW7eyvPe92/h1STtxLm8p7e0dIhEAAOBlnG4KAABAQSQCAABQEIkAAAAURCIAAAAFkQgAAEBBJAIAAFAQiQAAABREIgAAAAWRCAAAQEEkAgAAUBCJAAAAFEQiAAAABZEIAABAQSQCAABQEIkAAAAUykv1xr/97W9z/fXXF4+feeaZHHvssfn85z+fqVOnZtOmTRkxYkTOPffcJMkjjzySiRMnZsOGDamtrc3kyZNTXl6epUuXZvz48Vm1alX23XffTJs2Le9617uydu3a/OM//mMWL16cPn36ZObMmamuri7VcgAAAHYJJTuSOGbMmMyZMydz5szJtGnT0rdv33z961/PhAkTMmvWrDQ2Nubhhx/OXXfdlSQZP358Lrjggtxxxx3p6OjI7NmzkySTJ0/O2LFj09TUlEGDBmXWrFlJkpkzZ6a2tja33357xowZkylTppRqKQAAALuMnXK66Xe/+92ce+65Wbx4cfbZZ5/stddeKS8vz6hRo9LU1JQlS5Zk48aNGTx4cJJk9OjRaWpqSktLSxYsWJBhw4Z1Gk+SefPmZdSoUUmSkSNH5u67705LS8vOWA4AAMA7VslON33J/Pnzs3HjxowYMSJz587tdEpoTU1Nli9fnhUrVnQar66uzvLly7NmzZpUVVWlvLy803iSTtuUl5enqqoqq1evTr9+/bZrXn37Vr3qc5s2bkplz8odXuvb2a64ZgAA4JVKHok33HBDvvzlLydJ2tvbU1ZWVjzX0dGRsrKyVx1/6eOWXv54y226ddv+A6OrVq1Pe3vHVp+rru6VffsN2u73eidYtPzhNDev6+ppAAAAJdatW9k2D5qV9HTTzZs3Z8GCBfnc5z6XJOnfv3+am5uL55ubm1NTU/OK8ZUrV6ampiZ9+vTJunXr0tbW1un1yYtHIVeuXJkkaW1tzYYNG9K7d+9SLgcAAOAdr6SR+Oijj+b9739/dt999yTJIYcckkWLFuWpp55KW1tb5s6dm7q6ugwcODCVlZV54IEHkiRz5sxJXV1dKioqUltbm8bGxiRJQ0ND6urqkiRDhw5NQ0NDkqSxsTG1tbWpqKgo5XIAAADe8Up6uunixYvTv3//4nFlZWW+//3v56yzzsqmTZsydOjQDB8+PEkybdq0TJo0KevXr89BBx2Uk046KUly4YUXpr6+PldffXUGDBiQ6dOnJ0nOOeec1NfX5+ijj06vXr0ybdq0Ui4FAABgl1DW0dGx9Qvz3uFck9iZaxIBAGDX0KXXJAIAAPD2IhIBAAAoiEQAAAAKIhEAAICCSAQAAKAgEgEAACiIRAAAAAoiEQAAgIJIBAAAoCASAQAAKIhEAAAACiIRAACAgkgEAACgIBIBAAAoiEQAAAAKIhEAAICCSAQAAKAgEgEAACiIRAAAAAoiEQAAgIJIBAAAoCASAQAAKIhEAAAACiIRAACAgkgEAACgIBIBAAAoiEQAAAAKIhEAAICCSAQAAKAgEgEAACiIRAAAAAoiEQAAgIJIBAAAoCASAQAAKIhEAAAACiIRAACAgkgEAACgIBIBAAAoiEQAAAAKIhEAAICCSAQAAKAgEgEAACiIRAAAAAoiEQAAgIJIBAAAoCASAQAAKIhEAAAACiIRAACAgkgEAACgIBIBAAAolDQS//jHP2b06NEZMWJELrnkkiTJ/PnzM2rUqBx55JGZMWNG8dpHHnkko0ePzrBhwzJx4sS0trYmSZYuXZpx48Zl+PDhOeOMM7Jhw4Ykydq1a3PqqadmxIgRGTduXJqbm0u5FAAAgF1CySJx8eLFufDCCzNr1qzceuut+dvf/pa77rorEyZMyKxZs9LY2JiHH344d911V5Jk/PjxueCCC3LHHXeko6Mjs2fPTpJMnjw5Y8eOTVNTUwYNGpRZs2YlSWbOnJna2trcfvvtGTNmTKZMmVKqpQAAAOwyShaJv//973PUUUelf//+qaioyIwZM7Lbbrtln332yV577ZXy8vKMGjUqTU1NWbJkSTZu3JjBgwcnSUaPHp2mpqa0tLRkwYIFGTZsWKfxJJk3b15GjRqVJBk5cmTuvvvutLS0lGo5AAAAu4TyUr3xU089lYqKipx++ulZtmxZDj/88Oy///6prq4uXlNTU5Ply5dnxYoVncarq6uzfPnyrFmzJlVVVSkvL+80nqTTNuXl5amqqsrq1avTr1+/7Zpf375Vb9ZS3zGqq3t19RQAAIAuVrJIbGtry/3335/rrrsuu+++e84444z07NkzZWVlxWs6OjpSVlaW9vb2rY6/9HFLL3+85Tbdum3/gdFVq9anvb1jq8/tqrHU3Lyuq6cAAACUWLduZds8aFay00332GOPDBkyJH369EnPnj3z+c9/PvPnz+90g5nm5ubU1NSkf//+ncZXrlyZmpqa9OnTJ+vWrUtbW1un1ycvHoVcuXJlkqS1tTUbNmxI7969S7UcAACAXULJIvGII47In//856xduzZtbW3505/+lOHDh2fRokV56qmn0tbWlrlz56auri4DBw5MZWVlHnjggSTJnDlzUldXl4qKitTW1qaxsTFJ0tDQkLq6uiTJ0KFD09DQkCRpbGxMbW1tKioqSrUcAACAXUJZR0fH1s+5fBPceOON+fnPf56WlpYceuihmTRpUu67775MnTo1mzZtytChQ3P++eenrKwsCxcuzKRJk7J+/focdNBBmTp1anr06JElS5akvr4+q1atyoABAzJ9+vS85z3vybPPPpv6+vosXrw4vXr1yrRp0/K+971vu+f2Wqeb7ttv0Ju1G94WFi1/2OmmAACwC3it001LGolvZSKxM5EIAAC7hi67JhEAAIC3H5EIAABAQSQCAABQEIkA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bzNgwIAMGTIkN998c5Kkvb09ZWVlxWs6OjpSVlZWfNzSyx9vuU23bt1esc1L4zti1ar1aW/v2Opzu2osNTev6+opAAAAJdatW9k2D5qVJBIbGxvT3NycY489Ns8991yef/75lJWVFTeeSZKVK1empqYmffr0ybp169LW1pbu3bunubk5NTU1SZKampqsXLky/fv3T2trazZs2JDevXunX79+WbFiRfbee+9O7wUAAMAbU5JrEn/2s59l7ty5mTNnTs4+++x87nOfy9SpU1NZWZkHHnggSTJnzpzU1dWloqIitbW1aWxsTJI0NDSkrgVADTYAACAASURBVK4uSTJ06NA0NDQkeTE8a2trU1FRkaFDh2bOnDlJkvvvvz+VlZU7dKopAAAAW7dTf0/itGnTMnXq1AwfPjzPP/98TjrppCTJhRdemNmzZ+eoo47K/fffn2984xtJknPOOSd//etfc/TRR+fXv/51LrjggiTJiSeemM2bN+foo4/OlClTctlll+3MZQAAALxjlXV0dGz9wrx3uNe6JnHffoN28oy61qLlD7smEQAAdgGvdU3idh1JnDBhwivGzj777Nc/KwAAAN6StnnjmgsvvDDLly/PAw88kNWrVxfjra2tWbx4ccknBwAAwM61zUg8/vjj8/jjj+fRRx8tftl9knTv3j2DBw8u+eQAAADYubYZiQcffHAOPvjgfOYzn0n//v131pwAAADoItv1exKXLVuW8ePH57nnnsuW97m57bbbSjYxAAAAdr7tisQLLrggo0ePzoc//OGUlZWVek4AAAB0ke2KxPLy8nz5y18u9VwAAADoYtv1KzD233//PProo6WeCwAAAF1su44kLl68OF/84hez5557prKyshh3TSIAAMA7y3ZF4rnnnlvqeQAAAPAWsF2ReMABB5R6HgAAALwFbFckfvrTn05ZWVk6OjqKu5tWV1fn7rvvLunkAAAA2Lm2KxIXLlxY/Hnz5s2ZO3duFi1aVLJJAQAA0DW26+6mW+rRo0dGjx6de+65pxTzAQAAoAtt15HEZ599tvhzR0dHHn744axdu7ZkkwIAAKBr7PA1iUnSt2/fTJw4saQTAwAAYOfb4WsSAQAAeOfarkhsb2/PP//zP+fuu+9Oa2trDj300Jx++ukpL9+uzQEAAHib2K4b1/zTP/1T/vKXv+Tkk0/Ol7/85fz7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\n",
      "text/plain": [
       "<Figure size 1080x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set(rc={'figure.figsize':(15,8)})\n",
    "sns.countplot(loans['emp_length'],palette='rocket')\n",
    "plt.xlabel(\"Length\")\n",
    "plt.ylabel(\"Count\")\n",
    "plt.title(\"Distribution of Employement Length For Issued Loans\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can safely say that the majority of loans issued from Lending Club are from borrowers employed 10+ years so we will convert `emp_length` into categorical float values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "# convert `emp_length` into categorical float values\n",
    "mapping_dict = {\n",
    "    \"emp_length\": {\n",
    "        \"10+ years\": 10,\n",
    "        \"9 years\": 9,\n",
    "        \"8 years\": 8,\n",
    "        \"7 years\": 7,\n",
    "        \"6 years\": 6,\n",
    "        \"5 years\": 5,\n",
    "        \"4 years\": 4,\n",
    "        \"3 years\": 3,\n",
    "        \"2 years\": 2,\n",
    "        \"1 year\": 1,\n",
    "        \"< 1 year\": 0,\n",
    "        \"n/a\": 0\n",
    "    }\n",
    "}\n",
    "\n",
    "loans = loans.replace(mapping_dict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "# encode selected columns into dummy variables\n",
    "cat_columns = [\"home_ownership\", \"verification_status\", \"purpose\", \"term\"]\n",
    "dummy_df = pd.get_dummies(loans[cat_columns])\n",
    "loans = pd.concat([loans, dummy_df], axis=1)\n",
    "loans = loans.drop(cat_columns, axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "loan_amnt                              float64\n",
       "int_rate                               float64\n",
       "installment                            float64\n",
       "emp_length                              object\n",
       "annual_inc                             float64\n",
       "loan_status                              int64\n",
       "dti                                    float64\n",
       "delinq_2yrs                            float64\n",
       "fico_range_low                         float64\n",
       "fico_range_high                        float64\n",
       "inq_last_6mths                         float64\n",
       "open_acc                               float64\n",
       "pub_rec                                float64\n",
       "revol_bal                              float64\n",
       "revol_util                             float64\n",
       "total_acc                              float64\n",
       "last_fico_range_high                   float64\n",
       "last_fico_range_low                    float64\n",
       "collections_12_mths_ex_med             float64\n",
       "acc_now_delinq                         float64\n",
       "chargeoff_within_12_mths               float64\n",
       "delinq_amnt                            float64\n",
       "tax_liens                              float64\n",
       "home_ownership_MORTGAGE                  uint8\n",
       "home_ownership_NONE                      uint8\n",
       "home_ownership_OTHER                     uint8\n",
       "home_ownership_OWN                       uint8\n",
       "home_ownership_RENT                      uint8\n",
       "verification_status_Not Verified         uint8\n",
       "verification_status_Source Verified      uint8\n",
       "verification_status_Verified             uint8\n",
       "purpose_car                              uint8\n",
       "purpose_credit_card                      uint8\n",
       "purpose_debt_consolidation               uint8\n",
       "purpose_educational                      uint8\n",
       "purpose_home_improvement                 uint8\n",
       "purpose_house                            uint8\n",
       "purpose_major_purchase                   uint8\n",
       "purpose_medical                          uint8\n",
       "purpose_moving                           uint8\n",
       "purpose_other                            uint8\n",
       "purpose_renewable_energy                 uint8\n",
       "purpose_small_business                   uint8\n",
       "purpose_vacation                         uint8\n",
       "purpose_wedding                          uint8\n",
       "term_ 36 months                          uint8\n",
       "term_ 60 months                          uint8\n",
       "dtype: object"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "loans.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "loan_amnt                              0\n",
       "int_rate                               0\n",
       "installment                            0\n",
       "emp_length                             0\n",
       "annual_inc                             0\n",
       "loan_status                            0\n",
       "dti                                    0\n",
       "delinq_2yrs                            0\n",
       "fico_range_low                         0\n",
       "fico_range_high                        0\n",
       "inq_last_6mths                         0\n",
       "open_acc                               0\n",
       "pub_rec                                0\n",
       "revol_bal                              0\n",
       "revol_util                             0\n",
       "total_acc                              0\n",
       "last_fico_range_high                   0\n",
       "last_fico_range_low                    0\n",
       "collections_12_mths_ex_med             0\n",
       "acc_now_delinq                         0\n",
       "chargeoff_within_12_mths               0\n",
       "delinq_amnt                            0\n",
       "tax_liens                              0\n",
       "home_ownership_MORTGAGE                0\n",
       "home_ownership_NONE                    0\n",
       "home_ownership_OTHER                   0\n",
       "home_ownership_OWN                     0\n",
       "home_ownership_RENT                    0\n",
       "verification_status_Not Verified       0\n",
       "verification_status_Source Verified    0\n",
       "verification_status_Verified           0\n",
       "purpose_car                            0\n",
       "purpose_credit_card                    0\n",
       "purpose_debt_consolidation             0\n",
       "purpose_educational                    0\n",
       "purpose_home_improvement               0\n",
       "purpose_house                          0\n",
       "purpose_major_purchase                 0\n",
       "purpose_medical                        0\n",
       "purpose_moving                         0\n",
       "purpose_other                          0\n",
       "purpose_renewable_energy               0\n",
       "purpose_small_business                 0\n",
       "purpose_vacation                       0\n",
       "purpose_wedding                        0\n",
       "term_ 36 months                        0\n",
       "term_ 60 months                        0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "loans.isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Prediction Models\n",
    "\n",
    "Our data shows a severe class imbalance, we have a ratio of 5.5 `Fully Paid` loans to every one `Charged Off` loans. This is generally good news as more loans are paid back then charged off. However, we want our model to be accurate so we will use a confusion matrix to help define our **error metric**. This diagram from and article from [towardsdatascience](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62) gives a picture of predicted values and actual values categorized into true positives, true negatives, false positives, and false negatives.\n",
    "\n",
    "![data1](https://miro.medium.com/max/712/1*Z54JgbS4DUwWSknhDCvNTQ.png)\n",
    "\n",
    "We describe predicted values as positive and negative and actual values as true and false.\n",
    "\n",
    "![data2](https://miro.medium.com/max/880/1*2lptVD05HarbzGKiZ44l5A.png)\n",
    "\n",
    "\n",
    "\n",
    "#### False Postives\n",
    "\n",
    "False positive rate is the number of false positives divided by the number of false positives plus the number of true negatives. This divides all the cases where we thought a loan would be paid off but it wasn't by all the loans that weren't paid off. We can use the formula:\n",
    "\n",
    "`fpr = fp / (fp + tn)`\n",
    "\n",
    "#### True Positives\n",
    "\n",
    "True positive rate is the number of true positives divided by the number of true positives plus the number of false negatives. This divides all the cases where we thought a loan would be paid off and it was by all the loans that were paid off. We can use this formula:\n",
    "\n",
    "`tpr = tp / (tp + fn)`\n",
    "\n",
    "\n",
    "We will be using sklearn tools and pipelines including `predict_proba` which will give our prediction probabilites in an array form. In our particular binary classification problem, the `predict_proba` will give the probability of 0 and the probability of 1 which will correspond to `fpr` and `tpr` which we will graph.\n",
    "\n",
    "Using the confusion matrix should ensure that our models are not overfitting or being thrown off by the class imbalance.\n",
    "\n",
    "\n",
    "## Models\n",
    "\n",
    "Let's instantiate our first models. In order to fit the machine learning models, we'll use the Scikit-learn library and pipelines to apply normalization or scaling procedures. We will run preliminary models with basic class balancing.\n",
    "\n",
    "### Evaluating models: AUROC \n",
    "\n",
    "To evaluate our model we will use the Area Under the Receiver Operating Characteristics or \"AUROC\" curve to check the performance of our model. The ROC curve is plotted with TPR against the FPR where TPR is on y-axis and FPR is on the x-axis which represents probability. The AUC is the area underneath this curve. Higher AUC signifies that our model is better at predicting negative cases as negative cases(true negatives) and predicting positive cases as positive cases(true positives). \n",
    "\n",
    "\n",
    "### Logistic regression:\n",
    "\n",
    "We will split the model into training and testing data then scale the data and insantiate our model through a pipeline."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "# split data into training and testing data\n",
    "\n",
    "X = loans.drop('loan_status', axis=1)\n",
    "y = loans['loan_status']\n",
    "\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import sklearn tools\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "# transform which scales between 1 and 0\n",
    "sc = MinMaxScaler()\n",
    "# instantiate model using a balance class weight\n",
    "lr = LogisticRegression(class_weight='balanced', random_state=1)\n",
    "# applies transformation on logistic regression model using a pipeline\n",
    "pipe_lr = Pipeline([('scaler', sc), ('lr', lr)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Shaun/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Pipeline(memory=None,\n",
       "         steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),\n",
       "                ('lr',\n",
       "                 LogisticRegression(C=1.0, class_weight='balanced', dual=False,\n",
       "                                    fit_intercept=True, intercept_scaling=1,\n",
       "                                    l1_ratio=None, max_iter=100,\n",
       "                                    multi_class='warn', n_jobs=None,\n",
       "                                    penalty='l2', random_state=1, solver='warn',\n",
       "                                    tol=0.0001, verbose=0, warm_start=False))],\n",
       "         verbose=False)"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# train model using the the transformed insantiated logistic regression model\n",
    "pipe_lr.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Evaluating logistic regression: AUROC "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "No Skill ROC-AUC score: 0.50\n",
      "Linear Regression ROC-AUC score: 0.89\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA3oAAAHiCAYAAAC++b5/AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAgAElEQVR4nOzdd3RU1d7G8e850zLpkISAVGmCNKUJSLH3hiICCiJFbKCoCPbrVVCKimKniL2hlxcVy7WjglIuFkQRqSKEVFKnnXPeP0ZzRcBrSTIpz2ct1mI4ZX47bMI82fvsbTiO4yAiIiIiIiK1hhnrAkRERERERKRiKeiJiIiIiIjUMgp6IiIiIiIitYyCnoiIiIiISC2joCciIiIiIlLLKOiJiIiIiIjUMu5YF/B35OeXYNvVa3eItLREcnOLY12G1FLqX1LZ1MekMql/SWVS/5LKVB37l2ka1KuXcMDjNTro2bZT7YIeUC1rktpD/Usqm/qYVCb1L6lM6l9SmWpa/9LUTRERERERkVpGQU9ERERERKSWUdATERERERGpZWr0M3r7Y1kR8vOziURCMXn/3btNbNuOyXvLgbndXurVy8DlqnVdXkRERERkH7XuU29+fjZxcfEkJDTEMIwqf3+32yQSUdCrThzHoaSkkPz8bNLTG8W6HBERERGRSlfrpm5GIiESEpJjEvKkejIMg4SE5JiN8oqIiIiIVLVaF/QAhTzZh/qEiIiIiNQltTLoiYiIiIiI1GUKepVo586f6Nu3OytXrtjrzwcNOp2dO3/6w/dZs2YV48ZdxIUXDuWCCwbz0EP3YVkWAFdccTFr1qza6/xvv/2Gu+66fa/ja9as4oorLv6bLRIRERERkZpAQa+Sud1upk+fSmlpyV+6PhQKcdttN3HrrXfwxBPP8fjjz7BlyxZeeeWlA17Trt2hTJly818tWUREREREarhat+rmb511ln+fPzvjjAijRoUpLYVhw/Y9PmRImCFDIuTmGoweHbfP8ZEjw5x1VuQPvX96egY9ehzBnDmzmTz5xn2OP/nkAt5++w1M06RHj15cdtkEXC5X+fFAIEBJSTGBQBkAHo+HK6+8hrKysr3uk5+fx4QJl3DxxZeRkJDIggWP8cADj/2hGkVEREREpHbRiF4VuOKKq/j88+X7TOFcvvwTPv74I+bNe4oFC55hx47tLF788l7nJCcnM3z4RYwadQEXXjiE2bNnkZOTQ+vWbcrPKSkpZtKkqxg16mL69TuqKpokIiIiIiLVWK0f0Vu8uOyAx+Ljf/94Wprzu8f/qISERCZPvonp06fy5JPPl//56tUrOe64E4mLi44annrqGbzxxuucc87gva6/8MLRnHnmOaxcuYKVKz/j2msnMHbsJQwePAyAmTPvpH79NAYMOOZv1yoiIiIiIjWfRvSqSM+evcqncP7CcfbeWN1xwLL2nhL69ddf8corL5Gamsrxx5/EDTfcyrRpM1iyZHH5OeefP4LU1FT+9a9FldsIERERERGpESo96BUXF3Paaafx448/7nNs/fr1nH322Zx44onceOONRCJ/7Lm3muqXKZy5uTkAdO3ag3feeYtgMEAkEmHp0iV07dp9r2uSk5NZsOAxvv9+Q/mffffdt7Rte0j56zZtDuGaa6bw+ONzyc7eXTWNERERERGRaqtSg94XX3zB0KFD2bJly36PT5o0iVtuuYW33noLx3F48cUXK7OcmPtlCmc4HAbgyCP70adPX0aPHsHw4YPJzGzIOeect9c1zZo158Ybb+Wuu25nyJCBDB16Nps3/8DEidftdV7Tps04++xzueeeGVXWHhERERERqZ4Mx3Gcyrr5jTfeyMCBA7nuuut48sknadKkSfmxHTt2cOGFF/LOO+8AsGrVKu6//36efPLJP3z/3NxibHvv8nft2krDhs0rpgF/gdttEonY//tEqXKx7hsVISMjiezsoliXIbWY+phUJvUvqUzqX1LOcSAUwgiHcPzx4HJh7CnAzMmGcAQjFIRgECMUItytB8TF4fp+A6716zAikei1lgW2TWDwUPD5qmX/Mk2DtLTEAx6v1MVYpk6desBju3fvJiMjo/x1RkYGWVlZf+r++2vY7t0mbndsHz2M9fvL/pmmSUZGUqzL+NtqQxukelMfk8qk/iWVSf0rxhwHAgEoKYn+2rMHMjOjv/bsgfffh2AQwuH//howAA45BLZuhQULotcHAtHzgkEYPx66doXPPoObb4aiIti1C1JTo9cvWAA9e8KiRXDxxdFrf70N2RdfQOfOcM80uOuufWvevBkyMmDuW3DjvluhXfnBUB57Jdqvalr/itmqm7ZtYxhG+WvHcfZ6/Ufsb0TPtu2YjqhpRK/6sm272v0k5s+q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mojUMAp68ock3DQZp159Sq+ZTPDsc2NdjoiISK3iOPD22y7uusvHunUu2rWzePzxMk45JRLr0kSkhlLQk99nWSTeMAn/4/MInnpG9H8ijeSJiIhUKMOAuXO9lJQYPPRQGQMHRnC5Yl2ViNRkCnpyYKEQKUPOxvvxRwRPOoXCh+Yq5ImIiFSQlStNZs3yMWtWgKZNHR58MED9+g4eT6wrE5HaQIuxyAH55z6C9+OPKJ1wNYVPPAd+f6xLEhERqfG++spk2DA/p56awFdfmWzaFP04lpmpkCciFUcjenJAgQtG4P5qLSU3/SPWpYiIiNR4tg2XXRbHK694SE11uOmmIKNHh0hIiHVlIlIbaURP9pFw+614PngPJyWVokcWxLocERGRGi07O/rYg2lCRobD1VcHWbmymAkTFPJEpPIo6Mleki4dQ/yce3F//12sSxEREanRdu0ymDzZx+GHJ7BmTfQj1+23B5kyJURKSoyLE5FaT1M3pVzcgrnEvfwioSP7UTbq4liXIyIiUiPl5hrcf7+Xxx/3EInABReEadzYiXVZIlLHKOgJAN43l5I05Roi7Tuw55mX0JrOIiIif144DMceG8+uXQbnnhvh2muDNG+ukCciVU9BTwCwWrbCatiIwsefgvj4WJcjIiJSY5SUwCuveDj//DAeD9xxR5C2bW3atrVjXZqI1GEKenVdJIKZk43V9hDyVn4JPl+sKxIREakRgkF48kkPs2d7yc42ad3apndvi9NOi8S6NBERLcZS1yXeMImUgadibvpBIU9EROQPiETg6ac99OqVwI03xtG2rc2rr5bSu7cV69JERMppRK8Oi589C//C+URatcZucXCsyxEREakRbBtmz/aSmekwe3Yp/ftbGEasqxIR2ZuCXh3lXr2ShGn/xGrSlPwPlkc39xEREZF9OA688YabhQs9PPFEGX4/vPpqKQ0bOgp4IlJt6dN9HZUw7Z8AFCxeqimbIiIi++E48P77Lk46KZ6RI/1s327y44/Rj06NGinkiUj1phG9Oqps5Gisps2wmzWPdSkiIiLVzp49MGKEn+XL3TRpYjN7dhmDB0dw65OTiNQQ+nZV1zgOGAah084kdOoZsa5GRESkWtm926BBA4fkZKhf3+HOOwNccEFYk19EpMbR1M26xLJIPeMkUoacDYGAnssTERH52XffmYwaFUfPnglk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4IKgdu2yqXbtkNLTudEKAKB0KvGumxMmTDj633w+n/bv368zzzwz8snikGPDV5KkYMZ5FicBAEjSkSNS795JWr/erho1QnryyUJde21AdrvVyQAAOD5bSU94//33NX78eOXl5alDhw665pprNGfOnGhkizvud9+WJHm79bA4CQDEtz17itZipqRINWqENHmyR2vW5KtXL0oeAKBsKLHozZw5U9dff72WL1+uhg0b6qOPPtLixYujkS3uBC6qq4KhdylUo6bVUQAgLn33nU39+iWoSZNk7d5tyDCkp5/26Oab/XK5rE4HAMCJK3bp5u9M01Tt2rX17LPPqkWLFkpJSTl6ph7Cy3tdb3mtDgEAceg//zE0ebJbixY5VK6cdPfdPpUvz9c6AEDZVWLRs9lsWrp0qT755BONGjVKq1atksHtxcLOyDsi+/ffK3jhhTLLlbc6DgDEjexsQ1ddlSyHQ7rrLp+GDPEpNdXqVAAAnJ4Sl26OGjVK8+fP1z333KO0tDQ9/fTTGjt2bDSyxRXnp5+oYue2cqz/3OooABDzsrIMzZ3rlCSlpZl64gmP1q3L1wMPUPIAALGhxBm9Ro0aafbs2dq1a5d++eUXvfrqq9HIFXcc32yQJIWq17A4CQDErtxc6V//cunf/3bJ75datQrorLNM9ewZsDoaAABhVWLR2759u4YOHap9+/YpFAqpYsWKmjlzps4999xo5IsbzjWfSpKC551vcRIAiD15edLTT7v0zDMu5eVJPXoENHKkV2edxT48AEBsKnHp5vjx43Xrrbdq/fr1+vLLLzV48GD9/e9/j0a2uGJ4PQpWP1uylfi/BABwkrxeQ08/7dKVVwa0cmWBnn7ao4wMSh4AIHaV2CpycnLUvXv3o4979uypAwcORDRUPLLv+EX+yzOtjgEAMcHnk/79b6f69UuQaUqVK5tauzZfs2d7dOGFIavjAQAQcSUu3QwGgzp48KBSf9udnpubG/FQ8Sj386+lEN98AMDpCASk1193aMoUt3791abMzIAOHJAqVZLS05nBAwDEjxKL3o033qhevXrp6quvlmEYWrp0qf76179GI1tc4UgFADg9W7cauvHGRP38s10XXxzUo48WqFWroDgRCAAQj0oser169VLNmjX1ySefKBQKady4cWrWrFk0ssWNhOeekfuD5Tr03ItSSorVcQCgzDBN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\n",
      "text/plain": [
       "<Figure size 1080x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# predict probabilites and keeps only positive outcomes\n",
    "test_probas_lr = pipe_lr.predict_proba(X_test)[:,1]\n",
    "\n",
    "# import error metric tools\n",
    "from sklearn.metrics import roc_auc_score\n",
    "from sklearn.metrics import roc_curve\n",
    "\n",
    "# generate a no skill prediction\n",
    "ns_probs = [0 for _ in range(len(y_test))]\n",
    "\n",
    "# calculate scores\n",
    "ns_auc = roc_auc_score(y_test, ns_probs)\n",
    "lr_auc = roc_auc_score(y_test, test_probas_lr)\n",
    "\n",
    "print('No Skill ROC-AUC score: %.2f' % ns_auc)\n",
    "print('Linear Regression ROC-AUC score: %.2f' % lr_auc)\n",
    "\n",
    "# calculate roc curves\n",
    "ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)\n",
    "lr_fpr, lr_tpr, _ = roc_curve(y_test, test_probas_lr)\n",
    "\n",
    "# plot the roc curve for the model\n",
    "plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', color='blue')\n",
    "plt.plot(lr_fpr, lr_tpr, linestyle='--', label=\"Logistic Regression\", color='red')\n",
    "\n",
    "# axis labels\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "# show the legend\n",
    "plt.legend()\n",
    "# show the plot\n",
    "plt.show()\n",
    "\n",
    "# # calculate roc curve for false positive, true positive, and threshold rates\n",
    "# fpr, tpr, tresholds = roc_curve(y_test, test_probas_lr)\n",
    "# plt.plot(fpr, tpr, linestyle='--', label='')\n",
    "# plt.title('ROC')\n",
    "# plt.xlabel('FPR')\n",
    "# plt.ylabel('TPR')\n",
    "\n",
    "# print('Linear Regression ROC-AUC score: %.2f' % roc_auc_score(y_test, test_probas_lr))\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Logistic regression seems to be pretty accurate with a ROC-AUC-score of around 89%. Let's try some other models.\n",
    "\n",
    "### Random Forst Classifer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "#instantiate random forest classifier with a balanced class weight\n",
    "rfc = RandomForestClassifier(class_weight='balanced', random_state=1)\n",
    "\n",
    "pipe_rfc = Pipeline([('scaler', sc), ('rfc', rfc)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Shaun/opt/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Pipeline(memory=None,\n",
       "         steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),\n",
       "                ('rfc',\n",
       "                 RandomForestClassifier(bootstrap=True, class_weight='balanced',\n",
       "                                        criterion='gini', max_depth=None,\n",
       "                                        max_features='auto',\n",
       "                                        max_leaf_nodes=None,\n",
       "                                        min_impurity_decrease=0.0,\n",
       "                                        min_impurity_split=None,\n",
       "                                        min_samples_leaf=1, min_samples_split=2,\n",
       "                                        min_weight_fraction_leaf=0.0,\n",
       "                                        n_estimators=10, n_jobs=None,\n",
       "                                        oob_score=False, random_state=1,\n",
       "                                        verbose=0, warm_start=False))],\n",
       "         verbose=False)"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# train model using the the transformed insantiated random forest model\n",
    "pipe_rfc.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "No Skill ROC-AUC score: 0.50\n",
      "Linear Regression ROC-AUC score: 0.89\n",
      "Random Forest Classifier ROC-AUC score: 0.85\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# predict probabilites and keep only positive outcomes\n",
    "test_probas_rfc = pipe_rfc.predict_proba(X_test)[:,1]\n",
    "\n",
    "\n",
    "# calculate scores\n",
    "rfc_auc = roc_auc_score(y_test, test_probas_rfc)\n",
    "\n",
    "print('No Skill ROC-AUC score: %.2f' % ns_auc)\n",
    "print('Linear Regression ROC-AUC score: %.2f' % lr_auc)\n",
    "print('Random Forest Classifier ROC-AUC score: %.2f' % rfc_auc)\n",
    "\n",
    "# calculate roc curves\n",
    "rfc_fpr, rfc_tpr, _ = roc_curve(y_test, test_probas_rfc)\n",
    "\n",
    "# plot the roc curve for the model\n",
    "plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')\n",
    "plt.plot(lr_fpr, lr_tpr, linestyle='--', label=\"Logistic Regression\", color='red')\n",
    "plt.plot(rfc_fpr, rfc_tpr, linestyle='--', label=\"Random Forest Classifier\", color='green')\n",
    "\n",
    "\n",
    "# axis labels\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "# show the legend\n",
    "plt.legend()\n",
    "# show the plot\n",
    "plt.show()\n",
    "\n",
    "\n",
    "# fpr, tpr, tresholds = roc_curve(y_test, test_probas_rfc)\n",
    "# plt.plot(fpr, tpr)\n",
    "# plt.title('ROC')\n",
    "# plt.xlabel('FPR')\n",
    "# plt.ylabel('TPR')\n",
    "\n",
    "# print('Random Forest ROC-AUC score: %.2f' % roc_auc_score(y_test, test_probas_rfc))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Random forest classification scored less than logistic regression. Let's try one more model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Multinomial Naive Bayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.naive_bayes import MultinomialNB\n",
    "\n",
    "#instantiate multinomial naive bayes\n",
    "nb = MultinomialNB()\n",
    "\n",
    "pipe_nb = Pipeline([('scaler', sc), ('nb', nb)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(memory=None,\n",
       "         steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),\n",
       "                ('nb',\n",
       "                 MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))],\n",
       "         verbose=False)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# train model using the the transformed insantiated naive bayes model\n",
    "pipe_nb.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'X_ttest' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-42-b110b0d276a9>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# predict probabilites and keep only positive outcomes\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mtest_probas_nb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpipe_nb\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpredict_proba\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_ttest\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;31m# calculate scores\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'X_ttest' is not defined"
     ]
    }
   ],
   "source": [
    "# predict probabilites and keep only positive outcomes\n",
    "test_probas_nb = pipe_nb.predict_proba(X_ttest)[:,1]\n",
    "\n",
    "\n",
    "# calculate scores\n",
    "nb_auc = roc_auc_score(y_test, test_probas_nb)\n",
    "\n",
    "print('No Skill ROC-AUC score: %.2f' % ns_auc)\n",
    "print('Linear Regression ROC-AUC score: %.2f' % lr_auc)\n",
    "print('Random Forest Classifier ROC-AUC score: %.2f' % rfc_auc)\n",
    "print('Multinomial Naive Bayes ROC-AUC score: %.2f' % nb_auc)\n",
    "\n",
    "\n",
    "\n",
    "# calculate roc curves\n",
    "nb_fpr, nb_tpr, _ = roc_curve(y_test, test_probas_nb)\n",
    "\n",
    "# plot the roc curve for the model\n",
    "plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')\n",
    "plt.plot(lr_fpr, lr_tpr, linestyle='--', label=\"Logistic Regression\", color='red')\n",
    "plt.plot(rfc_fpr, rfc_tpr, linestyle='--', label=\"Random Forest Classifier\", color='green')\n",
    "plt.plot(nb_fpr, nb_tpr, linestyle='--', label='Naive Bayes', color='purple')\n",
    "\n",
    "\n",
    "# axis labels\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "# show the legend\n",
    "plt.legend()\n",
    "# show the plot\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Support Vector Machine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# from sklearn.svm import SVC\n",
    "\n",
    "# #instantiate support vector machine\n",
    "# svm = SVC(kernel = 'linear', random_state = 1)\n",
    "\n",
    "# pipe_svm = Pipeline([('scaler', sc), ('svm', svm)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# # train model using the the transformed insantiated support vector machine model\n",
    "# pipe_svm.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# # predict probabilites and keep only positive outcomes\n",
    "# test_probas_svm = pipe_svm.predict_proba(X_test)[:,1]\n",
    "\n",
    "\n",
    "# # calculate scores\n",
    "# svm_auc = roc_auc_score(y_test, test_probas_svm)\n",
    "\n",
    "# print('No Skill ROC-AUC score: %.2f' % ns_auc)\n",
    "# print('Linear Regression ROC-AUC score: %.2f' % lr_auc)\n",
    "# print('Random Forest Classifier ROC-AUC score: %.2f' % rfc_auc)\n",
    "# print('Multinomial Naive Bayes ROC-AUC score: %.2f' % nb_auc)\n",
    "# print('Support Vector Machine ROC-AUC score: %.2f' %svm_auc)\n",
    "\n",
    "\n",
    "\n",
    "# # calculate roc curves\n",
    "# svm_fpr, svm_tpr, _ = roc_curve(y_test, test_probas_svm)\n",
    "\n",
    "# # plot the roc curve for the model\n",
    "# plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')\n",
    "# plt.plot(lr_fpr, lr_tpr, linestyle='--', label=\"Logistic Regression\", color='red')\n",
    "# plt.plot(rfc_fpr, rfc_tpr, linestyle='--', label=\"Random Forest Classifier\", color='green')\n",
    "# plt.plot(nb_fpr, nb_tpr, linestyle='--', label='Naive Bayes', color='purple')\n",
    "# plt.plot(svm_fpr, svm_tpr, linestyle='--', label='Support Vector Machine', color='orange')\n",
    "\n",
    "\n",
    "# # axis labels\n",
    "# plt.xlabel('False Positive Rate')\n",
    "# plt.ylabel('True Positive Rate')\n",
    "# # show the legend\n",
    "# plt.legend()\n",
    "# # show the plot\n",
    "# plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## GridSearchCV\n",
    "\n",
    "Our Logistic Regression model scored the highest, so lets use a gridsearch to select better hyperparameters.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It seems multinomial naive bayes scored the lowest, most likely due to the class imbalance of `loan_status`, the high number of variables with different co\n",
    "\n",
    "# Conclusion\n",
    "\n",
    "Preliminarly we are able to create prediction models with decent accuracy. We should focus on logistic regression and random forest classification considering they scored higher with little modification. For logistic regression we could try issuing a penalty to the data to further help with our class imbalance, implement different class balancer ratio's again to help with class balancing, and feature engineering to combine certain columns that were dropped. For random forest classification we could change the number of estimators, change the leaf structure, the number of jobs, and create more categorical variables from previously dropped columns. Naive-Bayes may be the wrong model for this problem, it could be that each variable does not have strong independence from one another, which makes sense considering that borrowers with higher incomes or longer employment lengths are more likely to recieve loans with lower interest rates. \n",
    "\n",
    "We could also do some operations to the input data as far as removing outliers, refiling missing values with mean for columns that potentially have more correlation with the target variable, and creating new features. Creating new features in particular could help reduce the sheer amount of data as well. We could also try and find more correlations through a detailed analysis of LC's company history and connect this with general market trends for low capitol personal investments.\n",
    "\n",
    "The next steps are to refine the model, implement the model on current data, and create a preliminary investment plan. "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
