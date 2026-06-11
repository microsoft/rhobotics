## Launch a policy server

At a minimum, set the pretrained_checkpoint in ur5e_server.yaml to the policy you want to host, or override it from the command line. To load properly, you will need that `checkpoint.pt` to be available from the docker container (I usually host it under `/data/`) and don't forget to also download the `train_config.json` that is found at `../checkpoints`. BlobFusev2 should also work but I have not yet tested it.

```
cd environments/ur5e
python serve_policy.py --config_path new_server.yaml

OR

python serve_policy.py --config_path new_server.yaml --pretrained_checkpoint=<path/to/ckpt.pt>

```

Test your running server with

```
cd environments/ur5e
python dummy_client.py
```

Server with RTC:

```
cd environments/ur5e
python serve_policy.py --config_path new_server.yaml --policy_interface_cfg.eval_mode rtc --policy_interface_cfg.beta 10 --policy_interface_cfg.inference_delay 6

```

Test server with RTC:

```
cd environments/ur5e
python dummy_client.py

```

Test server with dataset client (launch server first):

```
python environments/ur5e/dataset_client.py --dataset_root /data/simran/bimanual_plug_0112/ --port 7000 --num_samples 300 --eval_mode rtc --beta 10 --inference_delay 8
```

## Common Errors

### client tries to connect, but policy server has not been launched

```
raise self.protocol.handshake_exc
websockets.exceptions.InvalidMessage: did not receive a valid HTTP response
```
probably forgot to launch the policy server, or it needs to be restarted.


### Missing websocket or rho_client -> reinstall rho client but it should already be in the `Dockerfile`

```
cd rho_client
pip install -e .
```

###
