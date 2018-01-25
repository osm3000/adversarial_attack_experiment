# adversarial_attack_experiment
This is an experiment I did in order understand and familiarize myself with adversarial attacks in machine learning. It is loosely inspired by the work done in [Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images](http://www.evolvingai.org/fooling). 

I decided to use evolutionary algorithms ([Evolutionary Strategies](http://www.scholarpedia.org/article/Evolution_strategies) and [Simple Genetic Algorithms](http://www.scholarpedia.org/article/Genetic_algorithms)) to perform the experiment. I used [Pygmo 2.7](https://esa.github.io/pagmo2/index.html). While it is an easy to use framework, I really don't like the parallelization toolbox in it (to be addressed later)

I present here my experimental setup and the results I obtained.
