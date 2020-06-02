---
layout: post
title:  "Policy Gradient Explained"
date:   2020-05-12 10:16:46 +0800
categories: reinforment_learning policy_gradient 
---
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    TeX: {
      equationNumbers: {
        autoNumber: "AMS"
      }
    },
    tex2jax: {
      inlineMath: [ ['$','$'], ['\(', '\)'] ],
      displayMath: [ ['$$','$$'] ],
      processEscapes: true,
    }
  });
</script>
<script type="text/javascript"
        src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# Introduction
This post explains the connection between various policy optimization algorithms in reinforcement learning.  

## policy optimization problem setup
The Reinforcement learning problem setup can be illustrated as a game tree. 

![game tree](/assets/images/game_tree.png)
{:.image-caption}
<center>figure 1. game tree</center>

A roll out of the game tree is as following:\
init state $s_1$ \
repeate untill game end:
1. agent choose a action $a_t$following policy function $\pi\(s_t\)$
2. following a unkonwn transition dynamics $p\(s_{t+1}\|s_t, a_t\)$, the environment reaches a new state $s_{t+1}$ 

Policy optimization algorithms aim at finding a optimal policy function $\pi$ that maximize the **expected return** if a agent plays the game following $\pi$. We use **"expected return"** because both $\pi$ and unkonwn transition fuction $p$ can be stochastic. And much of the following work focusing on find a efficient way to estimate the expectation. 

# Policy optimization algorithms
## Brute force search
A natual way of finding the optimal $\pi\(a|s\)$ is to exhausitive all possible policies, for each policy, estimate the **expeted return** by repeated play the game follwing the policy.

for each $\pi$ in $\Pi$: 
1. init $\pi^{opt}$, $r^{opt}$
2. play the game for X times and caculate the average return $r$, X need to be large enough for a accurate estimation
3. if $r>r^{opt}$: $\pi^{opt} \leftarrow \pi, r^{opt} \leftarrow r$ 

Brute force search is extreamly inefficienty. Because the number of possilbe policies $\|\Pi\|$ is very large. For a game with $M$ states and $N$ possible actions, the number of possible policies is $N^M$, the total number of demanded rollout would be $N^M \times X$, which is exponential.   	

## Monte Carlo Tree Search
The goal of our algorithm is find the **BEST** policy $\pi$, we don't care about the ranking of other policies. Therefore, as we carry out rollouts, we collect partial return distribution information over $\Pi$, we can utilize this information to attribute more computing power to more promising part of $\Pi$. This is the idea of Monte Carlo Tree Search algorithms.

We define two notations here:
1. The expeted value of taking action $a$ at state $s$: $q(s, a)$
2. the count of visits to $\(s, a\)$: $C\(s, a\)$

There two key steps in MCTS:\
repeate:
1. choose starting state $s_1$;
2. do a rollout following policy $a \sim \(softmax\(q\(s, a\)\) - softmax\(C\(s,a\)\)$\);
3. update $q\(s, a\), C\(s, a\)$ for all visited $\(s, a\)$ pair;

Step 2 is the core of MCTS, it visit more promising actions and main explore capacity by visit more "less visited actions", you can choose other exploit&explore strategies. 

## Vanila Policy Gradient
The idea of vanila policy gradient is very similar to MCTS except for one difference: use function approximation $\pi_{\theta}\(s\)$ to replace $\(softmax\(q\(s, a\)\)$. There are two benifit of this difference:
1. no book keeping of $q\(s, a\)$ needed, this is especially important when $\|\Pi\|$ is large to fit to memory; the parameterized approach also helps generalizatoin; 
2. gradient based update is more efficient in many situations. 

Policy gradient methods use a parameterized function $\pi_{\theta}\(a_t\|s_t\)$ to represent the policy, and turn the reinforcement learning problem into a numeric optimization problem. Therefore, finding the loss function is the core of policy gradient methods.

The target of reinforment learning is to maximize the expected return, which is a function of policy: $\mathop{\mathbb{E}}_{\theta}\[R\(\pi\_{\theta})\]$. However, $ \mathop{\mathbb{E}}\_{\theta}\[R\(\pi\_{\theta})\]$ can not be directly optimized because we do not have a direct function of $R\(\pi\_{\theta}\)$. What we can do is to do a rollout with a fixed policy $\pi\_{\theta'}$ and estimate  $ \mathop{\mathbb{E}}\_{\theta}\[R\(\pi\_{\theta})\]$ with importance sampling. 

$ \mathop{\mathbb{E}}\_{\theta}\[R\(\pi\_{\theta})\] = 
\frac
{\prod\_{t'=1}^{N}\pi\_{\theta}\(a\_{t'}\|s\_{t'})p(s\_{t+1}\|s\_t, a\_t\)}
{\prod\_{t'=1}^{N}\pi\_{\theta'}\(a\_{t'}\|s\_{t'})p(s\_{t+1}\|s\_t, a\_t\)}
\sum\_{t'=1}^Nr_{\theta'}\(s\_t, a\_t\)$
 
$= 
\frac{\prod\_{t'=1}^{N}\pi\_{\theta}\(s\_{t'})}
{ \prod\_{t'=1}^{N}\pi\_{\theta'}\(s_{t'}\)}
\sum\_{t'=1}^Nr_{\theta'}\(s\_t, a\_t\)$  // state transition probability is independent of $\theta$ 

The above loss function is still to complex to optimze. Vanila policy gradient simplify the function in two aspects:
1. update $\theta$ in each step $t'$ of the rollout;
2. aprroximate importance sampling factor only with the current step. 

The resulted approximation would be:\
for $t$ in $[1, N]$:
1. approximate $\mathop{\mathbb{E}}\_{\theta}\[R\(\pi\_{\theta})\]$ with 
$ 
\mathop{\mathbb{E}}\_{\theta} \[ \hat{R}\_t\(\pi\_{\theta}\) \]=
\frac{\pi\_{\theta}\(a\_t\|s\_{t})}
{\pi\_{\theta'}\(a\_t\|s_{t}\)}
\sum\_{t'=t}^Nr_{\theta'}\(s\_t, a\_t\)
$ 
by droping importance sampling factor of steps besides $t$
2. caculate gradient at $\theta'$:  
$
g_t=
\bigtriangledown\_{\theta}  \mathop{\mathbb{E}}\_{\theta} \[ \hat{R}\_t\(\pi\_{\theta}\) \] \| \theta'=
\bigtriangledown\_{\theta} \log \pi\_{\theta}\(a\_t\|s\_{t}\)\sum\_{t'=t}^Nr_{\theta'}\(s\_t, a\_t\)\|\theta' 
$

update $\theta$ with
$
\theta \leftarrow \theta + \alpha \times \sum\_{t=1}^Ng\_t 
$

In Vanila Policy Gradient algorithm, we caculate the gradient only with one rollout. Importance sampline enable us to make a unbiased estimate the distribution of data, with high variance. Much of the impovement over vanila policy gradient focus on reducing the variance. 

## Actor Critic
Vanila Policy Gradient requires finish the whole rollout to caculate the gradient of each time step. The idea of Actor Critic method a independent model(critic) to evaluate $\sum\_{t'=t}^Nr_{\theta'\(s\_t, a\_t\)}$ so that we can do $\theta$ update on each step of the rollout, and hope the following steps of the rollout have better policy, which imporve sampling efficiency. 

## Advantage Estimation 
Notice $G_t = \sum\_{t'=t}^Nr_{\theta'\(s\_t, a\_t\)}$ is not nomalized, which means the mean value of G_t is not zero. This problem result in high variance. The idea of Advantage Estimation is normalize $G_t$ with a baseline. Given a state, Advantage value evaluate the relative goodness of a action over the average performance follwing policy $\pi$. That is:\
$
A\(s\_t, a\_t\) = r\_{t+1} + V\_w\(s\_{t+1}) - V\_w\(s\_t\)
$
The gradient of loss function becomes:\
$
g_t=
\bigtriangledown\_{\theta} \log \pi\_{\theta}\(a\_t\|s\_{t}\)A\(s\_t, a\_t\) 
$

The Advantage Actor Critic has two main variants: the Asynchronous Advantage Actor Critic (A3C) and the Advantage Actor Critic (A2C). Both algorithms implement parallel training to reduce variance.

## Trust Region 
policy gradient method has high variance, one way to reduce oscillation is to limit the policy not changing too fast. Trust Region mechod limit the KL divergence between the new and the old policy, and turn the optimization problem into a constrained optimization problem:

$ 
\mathop{\mathbb{E}}\_{\theta}\[ 
\frac{\pi\_{\theta}\(a\_t\|s\_{t})}
{\pi\_{\theta'}\(a\_t\|s_{t}\)}
A\_w\(s\_t, a\_t\)
\]
$

s.t. 
$
\mathop{\mathbb{E}} \[
KL\[\pi\_{theta'}(\cdot\|s\_t\), \pi\_{theta}(\cdot\|s\_t\)\]\]
\leq
\lambda
$

We can solve this problem approximately using conjugate gradient. 

1. Convert constrained optimization to penaty: $\max\_{\theta}
\mathop{\mathbb{E}}\_{\theta}\[ 
\frac{\pi\_{\theta}\(a\_t\|s\_{t})}
{\pi\_{\theta'}\(a\_t\|s_{t}\)}
A\_w\(s\_t, a\_t\)
\]
-\beta \times
\mathop{\mathbb{E}} \[
KL\[\pi\_{\theta'}(\cdot\|s\_t\), \pi\_{\theta}(\cdot\|s\_t\)\]\]
$

2. Make linear approximation to 
$ 
\mathop{\mathbb{E}}\_{\theta}\[ 
\frac{\pi\_{\theta}\(a\_t\|s\_{t})}
{\pi\_{\theta'}\(a\_t\|s_{t}\)}
A\_w\(s\_t, a\_t\)
\]
$
and quadratic approximation to KL term:
$
\max\_{\theta} g \cdot \(\theta - \theta\_{old}\) - \frac{\beta}{2} \cdot \(\theta - \theta\_{old}\)^TF\(\theta - \theta\_{old}\) 
$
where $
g = \frac{\partial}{\partial \theta} L\_{\pi\_{\theta}'}\(\pi\_{\theta}\)\|\theta=\theta';
F = \frac{\partial^2}{\partial^2 \theta} KL\_{\pi\_{\theta}'}\(\pi\_{\theta}\)\|\theta=\theta'
$

3. update $\theta$: $\theta \leftarrow \theta' + \frac{1}{\beta}F^{-1}g$

There several limits of trust region method:
1. Actor and Critic cannot share architeture because we need to caculate KL divergence;
2. Empirically performs poorly on task requiring deep CNNs and RNNs;
3. Conjugate gradients makes implementation more complicated.

## Proximal Policy Optimization
PPO use approximate methods to limit the change of policy, but avoid using complex conjugate gradient.
### KL penaty version
This version use KL penaty as a penaty rather than constrain. The coefficient of KL penaty is dynamic, when KL divergence is high, it decreases $\beta$ at the next update, when KL divergence is low, it increses $\beta$ at the next update. 

### clipping objective
clipping objective method limit the policy change by clipping importance sampling ratio 
$r\_{\theta} = \frac{\pi\_{\theta}\(a\_n \| s\_n \)}{\pi\_{\theta'}\(a\_n \| s\_n \)}$
The loss function of lipping objective is:

$
L^{clicp}\_{\theta}=
\mathop{\mathbb{E}}\_{\theta}\[ 
min\(r\_t\(\theta\)A\_t, clicp\(r\_t\(\theta\), 1-\epsilon, 1+\epsilon\)A\_t\)\]
$

Cliping objective simplifies TRPO by removing KL penaty and works very well.

