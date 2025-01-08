# METASPACE

[The METASPACE platform](http://metaspace2020.eu/) hosts an engine for
 metabolite annotation of imaging mass spectrometry data as well as a
 spatial metabolite knowledgebase of the metabolites from thousands of
 public datasets provided by the community.

The METASPACE platform is developed by software engineers, data scientists and
 mass spectrometrists from the [Alexandrov team at UCSD](https://ateam.ucsd.edu/).
 This work is a part of the [European project METASPACE](https://cordis.europa.eu/project/id/634402).

## METASPACE in the CloudSkin project

This repository is a fork of the original METASPACE project, used as the metabolomics use-case for the [CloudSkin european project](https://cloudskin.eu/) (Grant agreement ID [101092646](https://doi.org/10.3030/101092646)). We integrate the METASPACE metabolomics annotation pipeline with a module for smart resource provisioning of cloud/edge resources. The module is a prototype that could potentially work as a part of the learning plane described in CloudSkin's reference architecture. The contributions in this repository were carried out during a research internship for the [CloudStars project](https://www.cloudstars.eu/) (Grant agreement ID 101086248) in IBM Research GmbH (Zurich, Switzerland).

## Contributions to the learning plane

The pipeline consists on several stages with different parallelism -i.e. number of parallel functions. The number of parallel functions is currently defined by partition size. However, our preliminary results show that only considering partition size for parallelism level might be suboptimal in terms of execution time. 

In [our prototype](metaspace/engine/sm/engine/learning_plane/) we implement an algorithm to choose the optimal parallelism with minimal overhead, considering CPU time, IO time and invocation overheads. We apply our algorithm to the sort operation of the input dataset, which happens to be critically influenced by the parallelism because of intermediate IO constraints.

Originally, the pipeline ran the sort locally in an available and properly configured AWS EC2 instance. We modify the pipeline code to run the sort operation distributedly on AWS Lambda instances, in case there is are not any configured AWS EC2 instances available.

For the learning plane to work, some profiling on the intermediate storage must be run beforeheand. The user has to setup its METASPACE and Lithops configuration correctly (see [the documentation](https://github.com/metaspace2020/metaspace/wiki/Project:-engine)) and then run:

```python
from sm.engine.config import SMConfig
from sm.engine.learning_plane.io.io_model import profile_storage

# Specify a correct path for you metaspace configuration file
SM_CONFIG_PATH = "<my-config-file" 
SMConfig.set_path(SM_CONFIG_PATH)

profile_storage()
```

## Projects

| Project | Description |
| :--- | :--- |
| [engine](metaspace/engine) | Contains a daemon that runs the metabolite annotation engine, and a REST API for sending jobs to the daemon |
| [graphql](metaspace/graphql) | A GraphQL API for accessing the annotations database and metabolite annotation engine |
| [webapp](metaspace/webapp) | A web application for submitting datasets and browsing results |
| [python-client](metaspace/python-client) | A Python library and set of example Jupyter notebooks for performing data analysis on the annotations database |
| [ansible](ansible) | Ansible playbooks for deploying to AWS |
| [docker](docker) | Docker Compose configuration for making development and testing environments |

Development documentation for each of these projects is available in the [wiki](https://github.com/metaspace2020/metaspace/wiki)

## Installation
Please check the [ansible](https://github.com/metaspace2020/metaspace/wiki/Ansible-server-provisioning-and-deployment)
documentation for production installations on AWS,
and the [docker](https://github.com/metaspace2020/metaspace/wiki/Docker-dev-environments)
documentation for development installations with Docker.

## Uploading Dataset and Browsing Results
Please visit the help page of our web application running on AWS:

[https://metaspace2020.eu/help](https://metaspace2020.eu/help)

## Acknowledgements
[<img src="https://user-images.githubusercontent.com/26366936/42039120-f008e4c6-7aec-11e8-97ea-87e48bf7bc1c.png" alt="BrowserStack" width="200">](https://www.browserstack.com)

METASPACE is tested with [BrowserStack](https://www.browserstack.com) to ensure cross-browser compatibility.
This service is provided for free under [BrowserStack's Open Source plan](https://www.browserstack.com/open-source).

[![Amazon Web Services and the “Powered by AWS” logo are trademarks of Amazon.com, Inc. or its affiliates in the United States and/or other countries.](https://d0.awsstatic.com/logos/powered-by-aws.png)](https://aws.amazon.com)

METASPACE is hosted on [Amazon Web Services](https://aws.amazon.com) with the support of [AWS Cloud Credits for Research](https://aws.amazon.com/research-credits/).

## Funding
We acknowledge funding from the following sources:

| | Funder / project(s) |
| :--- | :--- |
| <img src="https://metaspace2020.eu/img/Flag_of_Europe.80a3ee9f.svg" alt="EU" height="48" width="72"> | **European Union Horizon 2020 Programme** <br/> under grant agreements [634402](https://cordis.europa.eu/project/id/634402) / [773089](https://cordis.europa.eu/project/id/773089) / [825184](https://cordis.europa.eu/project/id/825184) |
| <img src="https://metaspace2020.eu/img/NIDDK.581b923e.svg" alt="EU" height="48" width="72"> | **National Institutes of Health NIDDK** <br/> Kidney Precision Medicine Project ([kpmp.org](https://kpmp.org/)) |
| <img src="https://metaspace2020.eu/img/NHLBI.6dbcd9a0.svg" alt="EU" height="48" width="72"> | **National Institutes of Health NHLBI** <br/> LungMAP Phase 2 ([lungmap.net](https://www.lungmap.net/)) |

and internal funds from the [European Molecular Biology Laboratory](https://www.embl.org/).

## License

Unless specified otherwise in file headers or LICENSE files present in subdirectories,
all files are licensed under the [Apache 2.0 license](LICENSE).
