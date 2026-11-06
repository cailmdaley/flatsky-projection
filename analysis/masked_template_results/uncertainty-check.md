# Paired uncertainty of residual/raw ratios

The final standard error of the masked residual fraction at ell900–1100 is **0.010 percentage points**, not the 0.005 percentage points obtained by holding the measured raw-BB denominator fixed.
For r=mean(y)/mean(x), the delta-method standard error is std(y-rx)/(sqrt(n) mean(x)); this retains numerator/denominator covariance and raw-leakage sample variance.
The deterministic 40,000-draw paired-seed bootstrap gives 0.00950 percentage points, compared with the delta method's 0.00979.
Its 95% interval for the residual percentage is [1.5395,1.5768].
The final quoted ratio is therefore **1.558 ±0.010%** of raw leakage.
The final summary and numbers exporter use this paired error, and `check_masked_template.py` gates both exact common-mode cancellation and a fluctuating-denominator example.

The archived independent final review reproduced the then-current fixed-denominator error; it is preserved as attributed review, not silently rewritten.
The ratio means, approximately twenty-fold cross-spectrum standard-deviation gain, paired bootstrap gain intervals, and all noise-limit conclusions are unchanged.
The revised paired error for the high-band Wiener cross bias is 0.38 percentage points, so that bias is 14.83±0.38% of raw leakage.

— Sol
