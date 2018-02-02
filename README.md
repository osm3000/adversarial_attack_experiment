# Adversarial Attack
This is an experiment I did in order understand and familiarize myself with adversarial attacks in machine learning. It is loosely inspired by the work done in [Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images](http://www.evolvingai.org/fooling).

I decided to use evolutionary algorithms ([Evolutionary Strategies](http://www.scholarpedia.org/article/Evolution_strategies) and [Simple Genetic Algorithms](http://www.scholarpedia.org/article/Genetic_algorithms)) to perform the experiment. I used [Pygmo 2.7](https://esa.github.io/pagmo2/index.html). While it is an easy to use framework, I really don't like the parallelization toolbox in it (to be addressed later)

I present here my experimental setup and the results I obtained.

## What is adversarial attack?
DISCLAIMER: A number of the following visualizations/explanations in this section are borrorwed from the presentation "Security and Privacy in Machine Learning", by Nicolas Papernot, Google Brain.

I will define an adversarial example attack in machine learning as introducing data examples that exploit the model limited knowledge about reality. The introduction of those examples can compromise the integrity of the predictions with respect to the expected outcome, and/or compromise the ability to deploy the system in real-life.

Machine learning is usually a part of a larger system.

Shape of the system:![](/presentation_images/attack_surface.png "Title if the picture")

Machine learning systems tries to approximate the real distribution based on the data samples provided. This is where the adversarial examples comes in.

Real distribution:![](/presentation_images/gen_er1.png)
Approximated distribution:![](/presentation_images/gen_er2.png)
Test data:![](/presentation_images/gen_er3.png)
Adversarial examples:![](/presentation_images/gen_er4.png)

## Hmm, let's see a demonstration then...
