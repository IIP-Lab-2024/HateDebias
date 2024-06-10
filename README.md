# HateDebias
Datasets and Codes for "HateDebias: On the Diversity and Variability of Hate Speech Debiasing"

# Data Construction

## Construct Our Corpus 
  ProcessData.py

# Static Debiasing

## Train with Static Method
  MTL{DebiasingMethod}.py

## Test with Static Method
  MTL{DebiasingMethod}-Test.py
(Except for the AT method, other debiasing methods can use the same test file)

# Continuous Debiasing

## Train with Continuous Method
  {ContinuousMethod}-{DebiasingMethod}.py

## Train with Continuous Method for Multiple Sequences at Once: 
  {ContinuousMethod}-{DebiasingMethod}-Search.py

## Test with Continuous Method
  {ContinuousMethod}-{DebiasingMethod}-Test.py
(Except for the AT method, other debiasing methods can use the same test file)
