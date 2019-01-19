# Geometric Neural Networks

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

TODO: rules


