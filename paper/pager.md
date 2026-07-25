---
title: Projection Paper Pager
author: Cail Daley
---

![Wallis Plot](wallis_plot.png)


<!-- ::: {style="text-align:center"} -->

![Maps](){#maps width=50%}

<!-- ::: -->

<script language="Javascript">
  opts = {
         'Figure Type|figtype' : [
           'Q Projected|projectedQU_Q', 
           'U Projected|projectedQU_U', 
           'E Projected|projectedEB_E', 
           'B Projected|projectedEB_B', 
           'E Flat-sky|projectedQU_E', 
           'B Flat-sky|projectedQU_B', 
           'E Residuals|residuals_E', 
           'B Residuals|residuals_B', 
         ],
         'Projection|proj' : [
           'ProjSFL', 
           'ProjSIN', 
           'ProjSTG', 
           'ProjZEA', 
           'ProjECP'
           ],
  };
  pager.link('#maps', opts,
      function(p){
         fn = `${p.figtype}_seed1_${p.proj}_vlim=dynamic.png`;
         return fn;
      }
  );
  pager.setparams({'figtype': 'projectedQU_Q', 
                   'proj' : 'ProjZEA'});
</script>