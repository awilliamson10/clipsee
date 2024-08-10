The vit_prisma library isn't able to correctly load the OpenCLIP models as they incorrectly provide the number of classes. So this requires a small patch to work correctly.

```python
# vit_prisma.prima_tools.loading_from_pretrained 
# line 338
hf_config = AutoConfig.from_pretrained(model_name)
hf_config.vision_config.projection_dim = hf_config.projection_dim
hf_config = hf_config.vision_config
```