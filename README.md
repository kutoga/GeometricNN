
# Geometric Neural Networks  

IMPORTANT: This project is not yet finished!

  
Classical neural networks are based on [classical calculus](https://en.wikipedia.org/wiki/Calculus) of Newton and Leibniz. One might ask if there are useful alternatives?  
  
## Why classical calculus?  
  
We mainly focus on derivatives: Classical calculus derivatives are additive:  
  
![mathematical expression](doc/img/6675f4c05945b4216da30a575c928b06.svg)  
  
This property is very helpful to differentiate linear functions like:  
![mathematical expression](doc/img/b348b4a6e76f3ced06eae2ec8c4c3496.svg)  
  
One can use different well-known optimization methods which are based on these derivatives.  
  
## "Non-classical" calculus?  
  
What even is this? [Wikipedia](https://en.wikipedia.org/wiki/List_of_derivatives_and_integrals_in_alternative_calculi) might help you, but if you read the next few lines, it might be even easier for you, to follow my idea.  
  
You probably already know the intergral operator: ![mathematical expression](doc/img/277fc11f85f539e0faaa5ec22a5e9489.svg). You probably know that this is somehow a continuous version of ![mathematical expression](doc/img/8c14b498ee4985136badc3dfa030932d.svg). These two operators are closely related. There is also a multiplication-operator ![mathematical expression](doc/img/0d76a5dcfaf65104c0fdd46679dfe235.svg) which computes a product over a discrete set of numbers. But, is there also a continous version of this operator? Actually, it is not that hard to derive a continuous version:  
  
![mathematical expression](doc/img/2de26b85a1cf69d4f79e3d5d3be909fe.svg)  
![mathematical expression](doc/img/5a0cf5cf56d5a1ba05a57d3e5e043e4b.svg)  
  
Now, we can replace the ![mathematical expression](doc/img/277207eeef2281959d3187e0bb79d2c1.svg) by the integral operator. This results in:  
  
![mathematical expression](doc/img/7da40b1144ddf92b4154e07dd664c10f.svg)  
  
Cool, we have a continuous product operator. There is a well-known syntax for this operator (which I actually do not like too much, but it is ok):  
  
![mathematical expression](doc/img/8873d9147a8547acefba4d7add80e0ad.svg)  
  
We can even remove the limits:  
  
![mathematical expression](doc/img/b76ad5c3e62414e73d0af733687bcbda.svg)  
  
This "new" integral-like operator has some interesting properties, probably the most important are:  
  
![mathematical expression](doc/img/425a193a73e548ef684dacf7b0939279.svg)  
  
![mathematical expression](doc/img/23a06af72f899f8612cb2774807ab776.svg)  
  
One might think about to invert this operator, which is actually quite easy. We denote the resulting expression as ![mathematical expression](doc/img/372b13f27289207d40f1fac233eb205a.svg):  
  
![mathematical expression](doc/img/e967e4493d54d1e95c0168f54099bec4.svg)  
  
This operator is also multiplicative and has another interesting property: It removes any constants from the input function:
  
![mathematical expression](doc/img/01381ac9312352f2700cd1467d14cd6b.svg)  
  
![mathematical expression](doc/img/2760bc166732e71b4716c97e0ce25447.svg)  
  
As it can be seen easily, this operator cannot handle functions with zeros: The operator is undefined for any position which results in a zero value. As it can be done for  
the "normal" derivative, we can derive the chain-rule and many other simple derivative rules for this operator. One might find the following derivative rules:  
  
![mathematical expression](doc/img/388d62ce0d545ac16062a523a8ae2643.svg)  
  
![mathematical expression](doc/img/d5fac6296d05c4ba98c40ab8f9be1a9c.svg)

A useful property is that the derivative is multiplicative: This allows a much more efficient computation of the multiplicative derivative of products (compared to the classical derivative), because each factor (=function) can be handled independent of each other.

<!--Using backpropagation, this results in a quite nice algorithm. Even better, by knowing the following rule, this new knowledge can also be used to compute classical gradients more efficient for products:

![mathematical expression](doc/img/163504d7efbe99a21795bd585853335b.svg)
![mathematical expression](doc/img/866eb7ed1611a3659362abfc95689348.svg)
![mathematical expression](doc/img/e967e4493d54d1e95c0168f54099bec4.svg)

How can this improve the computation of the classical gradient? This is actually a very nice trick. Given the following functions:

![mathematical expression](doc/img/72a8b9da2a400bef8befae62f5a28bbf.svg)

![mathematical expression](doc/img/30de9c3a43927c207a23970a707ffcfd.svg)

![mathematical expression](doc/img/1cee9c54587393af4b8668ba566a26e8.svg)

How can we compute the gradient of the function ![mathematical expression](doc/img/6e6638d593d032139132402aec60e526.svg) at a fixed position, given we evaluate the function at this position (forward pass in a neural network)? Of course, we can use the classical chain rule:
-->

The "classical" gradient descent is defined as:

![mathematical expression](doc/img/bfacbc7bb4b1f2284574285d58fc1bff.svg)

Which can be written as:

![mathematical expression](doc/img/ebe7f01c81de5946dfe7a263be0fbdeb.svg)

Of course, given a neural network, ![mathematical expression](doc/img/c310b14542d868fdb5f9652a389cf000.svg) is usually ![mathematical expression](doc/img/2ec6968afb38f142a5252dd886a0a27a.svg) and describes weights.

Very similar, we can define a multiplicative gradient descent:

![mathematical expression](doc/img/242da9d3097369b580a5b042ecde14a4.svg)

One might ask, what this "optimization" method does? Or even more basic: What is the geometric meaning of this multiplicative derivative? For the classical derivative,
we all know that it is the tangent of the function at a given point. It shows in which direction the function increases (a positive derivative means the function grows
with a higher ![mathematical expression](doc/img/c310b14542d868fdb5f9652a389cf000.svg) value and ![mathematical expression](doc/img/3d8e1f922d8c67053e34fe9b3de59ada.svg) means the function decreases). The multiplicative derivative on the other side, shows in which direction the absolute value of the function
increases: This means it show the direction away from ![mathematical expression](doc/img/7770202207bf90dfebebe54e52bf326a.svg). This is not given by the absolute value, but by a factor: If the multiplicative derivative is ![mathematical expression](doc/img/782bcbc35b39fd0e8164a3789751b96b.svg), then the function
will have a larger absolute value with a larger ![mathematical expression](doc/img/c310b14542d868fdb5f9652a389cf000.svg) and if it it ![mathematical expression](doc/img/e9496f293beae580549cda815aac75d3.svg), then the function will have a smaller absolute value with a larger ![mathematical expression](doc/img/c310b14542d868fdb5f9652a389cf000.svg).

The multiplicative gradient descent therefore divides by the multiplicative gradient. Unfortunately, the multiplicative gradient is independent of the absolute position of ![mathematical expression](doc/img/c310b14542d868fdb5f9652a389cf000.svg)
and for large ![mathematical expression](doc/img/c310b14542d868fdb5f9652a389cf000.svg) this division can create very large steps. For this reason, the proposed corrected multiplicative derivative is:

![mathematical expression](doc/img/b2b18fe56856d80ca745bd3d2eef2256.svg)

## What do you try to show?

I want to show that the multiplicative derivative can be useful to optimize neural networks. Especially multiplicative neural networks. I want to show this with a minimal library
that implements simple neural networks and which can be seen as a proof-of-concept that this (maybe?) new method works. Or that it does not work.

