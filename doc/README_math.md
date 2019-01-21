
# Geometric Neural Networks  
  
Classical neural networks are based on [classical calculus](https://en.wikipedia.org/wiki/Calculus) of Newton and Leibniz. One might ask if there are useful alternatives?  
  
## Why classical calculus?  
  
We mainly focus on derivatives: Classical calculus derivatives are additive:  
  
$f(x)=g(x)+h(x)\Rightarrow f'(x)=g'(x)+h'(x)$  
  
This property is very helpful to differentiate linear functions like:  
$f(x_0, x_1, x_2)=a x_0 + b x_1 + c x_2 + d$  
  
One can use different well-known optimization methods which are based on these derivatives.  
  
## "Non-classical" calculus?  
  
What even is this? [Wikipedia](https://en.wikipedia.org/wiki/List_of_derivatives_and_integrals_in_alternative_calculi) might help you, but if you read the next few lines, it might be even easier for you, to follow my idea.  
  
You probably already know the intergral operator: $\int\limits_a^b f(x) \mathrm{d}x$. You probably know that this is somehow a continuous version of $\sum\limits_a^b f(x)$. These two operators are closely related. There is also a multiplication-operator $\prod\limits_a^b f(x)$ which computes a product over a discrete set of numbers. But, is there also a continous version of this operator? Actually, it is not that hard to derive a continuous version:  
  
$\prod\limits_a^b f(x)=\exp\left(\log\left(\prod\limits_a^b f(x)\right)\right)$  
$\prod\limits_a^b f(x)=\exp\left(\sum\limits_a^b \log\left(f(x)\right)\right)$  
  
Now, we can replace the $\sum$ by the integral operator. This results in:  
  
$\exp\left(\int\limits_a^b \log\left(f(x)\right)\right)$  
  
Cool, we have a continuous product operator. There is a well-known syntax for this operator (which I actually do not like too much, but it is ok):  
  
$\prod\limits_a^b f(x)^{\mathrm{d}x}=\exp\left(\int\limits_a^b \log\left(f(x)\right)\right)$  
  
We can even remove the limits:  
  
$\prod\limits f(x)^{\mathrm{d}x}=\exp\left(\int\limits \log\left(f(x)\right)\right)$  
  
This "new" integral-like operator has some interesting properties, probably the most important are:  
  
$\prod\limits \left(c^{f(x)}\right)^{\mathrm{d}x}=c^{\int\limits f(x)\mathrm{d}x}$  
  
$\prod\limits \left(g(x)h(x)\right)^{\mathrm{d}x}=\left(\prod\limits g(x)^{\mathrm{d}x}\right)\left(\prod\limits h(x)^{\mathrm{d}x}\right)$  
  
One might think about to invert this operator, which is actually quite easy. We denote the resulting expression as $f^{*}(x)$:  
  
$f^{*}(x)=\exp\left(\frac{f'(x)}{f(x)}\right)$  
  
This operator is also multiplicative and has another interesting property: It removes any constants from the input function:  
  
$f(x)=g(x)h(x)\Rightarrow f^{*}(x)=g^{*}(x)h^{*}(x)$  
  
$f(x)=c g(x), c\neq 0 \Rightarrow f^{*}(x)=g^{*}(x)$  
  
As it can be seen easily, this operator cannot handle functions with zeros: The operator is undefined for any position which results in a zero value. As it can be done for  
the "normal" derivative, we can derive the chain-rule and many other simple derivative rules for this operator. One might find the following derivative rules:  
  
$f(x)=c^{g(x)}, c>0 \Rightarrow f^{*}(x)=c^{f'(x)}$  
  
$f(x)=g(h(x)) \Rightarrow f^{*}(x)=g^{*}(h(x))^{f(x)\log(f^{*}(x))}=g^{*}(h(x))^{f'(x)}$

A useful property is that the derivative is multiplicative: This allows a much more efficient computation of the multiplicative derivative of products (compared to the classical derivative), because each factor (=function) can be handled independent of each other.

<!--Using backpropagation, this results in a quite nice algorithm. Even better, by knowing the following rule, this new knowledge can also be used to compute classical gradients more efficient for products:

$f'(x)=f(x)\log(f^{*}(x))$
$\Leftrightarrow$
$f^{*}(x)=\exp\left(\frac{f'(x)}{f(x)}\right)$

How can this improve the computation of the classical gradient? This is actually a very nice trick. Given the following functions:

$g(x_0, x_1, x_2)=w_0^{x_0} w_1^{x_1} w_2^{x_2}b$

$h(x)=x^2$

$f(x_0, x_1, x_2)=h(g(x))$

How can we compute the gradient of the function $f$ at a fixed position, given we evaluate the function at this position (forward pass in a neural network)? Of course, we can use the classical chain rule:
-->
