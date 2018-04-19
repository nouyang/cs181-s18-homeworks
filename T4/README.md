On problem 2-1, we  are supposed to calculate $ p(X| \{\phi_k\})$ , where $phi_k$ gives the
parameters $\mu_k$ and $\Sigma_k$ for a component.

 

 We are instructed to include the necessary sums over observations $\{ x_i \}$ and latents $\{ z_i
 \}$. 

  

  What I don't understand is how we are supposed to sum over the latents because 

  1.) they don't appear anywhere in the expression we are supposed to calculate (they aren't in  $\{
  x_i \}$ or  $\{ \phi_i \}$

  2.) they are distributed according to $\theta,$ which we also haven't been given.

   

   Don't we need at least one of these to be given so we can condition on it? What am I missing?
==

   Yeah that seems like it should have a theta_k. You can include a theta_k in your expression. 
