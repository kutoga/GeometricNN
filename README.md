# Geometric Neural Networks

Classical neural networks are based on [classical calculus](https://en.wikipedia.org/wiki/Calculus) of Newton and Leibniz. One might ask if there are useful alternatives?

## Why classical calculus?

We mainly focus on derivatives: Classical calculus derivatives are additive:

![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20f%28x%29%3Dg%28x%29%2Bh%28x%29%5CRightarrow%20f%27%28x%29%3Dg%27%28x%29%2Bh%27%28x%29)

This property is very helpful to differentiate linear functions like:
![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20f%28x_0%2C%20x_1%2C%20x_2%29%3Da%20x_0%20%2B%20b%20x_1%20%2B%20c%20x_2%20%2B%20d)

One can use different well-known optimization methods which are based on these derivatives.

## "Non-classical" calculus?

What even is this? [Wikipedia](https://en.wikipedia.org/wiki/List_of_derivatives_and_integrals_in_alternative_calculi) might help you, but if you read the next few lines, it might be even easier for you, to follow my idea.

You probably already know the intergral operator: ![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20%5Cint%5Climits_a%5Eb%20f%28x%29%20%5Cmathrm%7Bd%7Dx). You probably know that this is somehow a continuous version of ![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20%5Csum%5Climits_a%5Eb%20f%28x%29). These two operators are closely related. There is also a multiplication-operator ![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20%5Cprod%5Climits_a%5Eb%20f%28x%29) which computes a product over a discrete set of numbers. But, is there also a continous version of this operator? Actually, it is not that hard to derive a continuous version:

![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20%5Cprod%5Climits_a%5Eb%20f%28x%29%3D%5Cexp%5Cleft%28%5Clog%5Cleft%28%5Cprod%5Climits_a%5Eb%20f%28x%29%5Cright%29%5Cright%29)
![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20%5Cprod%5Climits_a%5Eb%20f%28x%29%3D%5Cexp%5Cleft%28%5Csum%5Climits_a%5Eb%20%5Clog%5Cleft%28f%28x%29%5Cright%29%5Cright%29)

Now, we can replace the ![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20%5Csum) by the integral operator. This results in:

![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20%5Cexp%5Cleft%28%5Cint%5Climits_a%5Eb%20%5Clog%5Cleft%28f%28x%29%5Cright%29%5Cright%29)

Cool, we have a continuous product operator. There is a well-known syntax for this operator (which I actually do not like too much, but it is ok):

![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20%5Cprod%5Climits_a%5Eb%20f%28x%29%5E%7B%5Cmathrm%7Bd%7Dx%7D%3D%5Cexp%5Cleft%28%5Cint%5Climits_a%5Eb%20%5Clog%5Cleft%28f%28x%29%5Cright%29%5Cright%29)

We can even remove the limits:

![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20%5Cprod%5Climits%20f%28x%29%5E%7B%5Cmathrm%7Bd%7Dx%7D%3D%5Cexp%5Cleft%28%5Cint%5Climits%20%5Clog%5Cleft%28f%28x%29%5Cright%29%5Cright%29)

This "new" integral-like operator has some interesting properties, probably the most important are:

![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20%5Cprod%5Climits%20%5Cleft%28c%5E%7Bf%28x%29%7D%5Cright%29%5E%7B%5Cmathrm%7Bd%7Dx%7D%3Dc%5E%7B%5Cint%5Climits%20f%28x%29%5Cmathrm%7Bd%7Dx%7D)

![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20%5Cprod%5Climits%20%5Cleft%28g%28x%29h%28x%29%5Cright%29%5E%7B%5Cmathrm%7Bd%7Dx%7D%3D%5Cleft%28%5Cprod%5Climits%20g%28x%29%5E%7B%5Cmathrm%7Bd%7Dx%7D%5Cright%29%5Cleft%28%5Cprod%5Climits%20h%28x%29%5E%7B%5Cmathrm%7Bd%7Dx%7D%5Cright%29)

One might think about to invert this operator, which is actually quite easy. We denote the resulting expression as ![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20f%5E%7B%2A%7D%28x%29):

![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20f%5E%7B%2A%7D%28x%29%3D%5Cexp%5Cleft%28%5Cfrac%7Bf%27%28x%29%7D%7Bf%28x%29%7D%5Cright%29)

This operator is also multiplicative and has another interesting property: It removes any constants from the input function:

![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20f%28x%29%3Dg%28x%29h%28x%29%5CRightarrow%20f%5E%7B%2A%7D%28x%29%3Dg%5E%7B%2A%7D%28x%29h%5E%7B%2A%7D%28x%29)

![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20f%28x%29%3Dc%20g%28x%29%2C%20c%5Cneq%200%20%5CRightarrow%20f%5E%7B%2A%7D%28x%29%3Dg%5E%7B%2A%7D%28x%29)

As it can be seen easily, this operator cannot handle functions with zeros: The operator is undefined for any position which results in a zero value. As it can be done for
the "normal" derivative, we can derive the chain-rule and many other simple derivative rules for this operator. One might find the following derivative rules:

![mathematicla expression](http://latex.codecogs.com/png.download?%5Cinline%20%5Cdpi%7B120%7D%20%5Clarge%20f%28x%29%3Dc%5E%7Bg%28x%29%7D%2C%20c%3E0%20%5CRightarrow%20f%5E%7B%2A%7D%28x%29%3Dc%5E%7Bf%27%28x%29%7D)

TODO: rules


