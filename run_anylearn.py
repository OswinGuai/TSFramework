from anylearn import init_sdk, quick_train


init_sdk('http://anylearn.nelbds.cn', 'DigitalLifeYZQiu', 'Qyz20020318!')



for model in ['transformer']:
    for pattern in ['pretrain','train']:
        cmd = "sh ./scripts/{}_{}.sh".format(model,pattern)
        print(cmd)
        task, _, _, _ = quick_train(
                            project_name='TSFramework',
                            algorithm_cloud_name=f"TSFramework_{model}_{pattern}",
                            algorithm_local_dir="/workspace/qiuyunzhong/TSFramework/",
                            algorithm_entrypoint=cmd,
                            algorithm_force_update=True,
                            algorithm_output="./outputs",
                            dataset_id=["DSET924f39a246e2bcba76feef284556", "DSETecc2e54a4c80a793255c932e7b72", "DSET778fcf74414d8e186fd05350ebee", "DSETe9eb0a5a4b40876add2dbd3acb6a", "DSET73e1e542467986886113370b39d1", "DSET69dd739245f59853a74d98d2cc4c", "DSETb990ae96465d9eff1bfff43e5eca", "DSET8a91cb7146f58a081f7fe7561dea", "DSETffd84b7f4e4e81ad73db993d91e8"],
                            model_id=[ "MODE9542ece644e6a8016523aa882aca","MODE6c8b17084c97b580924c649d9ea8","MODE39fa73c14582b94a3b6091d86021","MODEc4815db3430da0d0db891187d2dc","MODE9c2965fa48e895cadef4febeaba8","MODEb6ab0e91432789fa4e6f7e102208","MODE5eaaa691455bae08dab073433837","MODE356bc22440ef999860540caa6fae"],
                            image_name="QUICKSTART_PYTORCH1.13.0_CUDA11.7",
                            quota_group_request={
                                'name': "DL2023",
                                'RTX-3090-shared': 1,
                                'CPU': 8,
                                'Memory': 30},
                        )
