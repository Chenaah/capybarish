# 🦫 Capybarish

[![PyPI version](https://badge.fury.io/py/capybarish.svg)](https://badge.fury.io/py/capybarish)
[![Python Support](https://img.shields.io/pypi/pyversions/capybarish.svg)](https://pypi.org/project/capybarish/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2505.00784-b31b1b.svg)](https://arxiv.org/abs/2505.00784)

**Capybarish** is a lightweight communication middleware designed for cloud-based robotics controllers. This MVP (Minimum Viable Product) was developed to support research on [reconfigurable legged metamachines](https://arxiv.org/abs/2505.00784) - modular robots that can dynamically reconfigure their morphology using autonomous modular legs.

## 🔬 Research Context

This package supports the research presented in ["Reconfigurable legged metamachines that run on autonomous modular legs"](https://arxiv.org/abs/2505.00784) by Chen Yu et al. The middleware enables real-time communication between cloud-based controllers and distributed modular robot components, facilitating the coordination of autonomous modular legs that can form complex legged metamachines.

## 🚀 Quick Start

### Installation

```bash
pip install capybarish
```

### Basic Usage

```python
from capybarish.interface import Interface
from capybarish.utils import load_cfg
import numpy as np

# Load configuration
config = load_cfg("default")

# Initialize the interface
interface = Interface(config)

# Basic control loop for modular robot communication
for i in range(100):
    # Receive sensor data from robot modules
    interface.receive_module_data()
    
    # Get observable data
    data = interface.get_observable_data()
    
    # Send control commands to modular legs
    action = np.array([0.1 * np.sin(i * 0.1)])
    interface.send_action(action, kps=np.array([10.0]), kds=np.array([1.0]))
```

### Configuration

Create a YAML configuration file in the `config/` directory:

```yaml
# config/default.yaml
interface:
  module_ids: [1, 2, 3]  # IDs of your modular robot components
  communication:
    protocol: "udp"
    timeout: 0.1
  dashboard:
    enabled: true
    port: 6667
```

## 🏗️ Architecture

Capybarish provides a modular middleware architecture with:

- **Interface**: Main control interface for robot communication
- **Communication Manager**: UDP-based real-time communication
- **Dashboard Server**: Data visualization and monitoring
- **Plugin System**: Extensible architecture for custom components
- **Configuration Management**: Flexible YAML-based setup

## 📋 Requirements

- Python 3.8+
- NumPy >= 1.20.0
- OmegaConf >= 2.1.0
- MessagePack >= 1.0.0
- Rich >= 10.0.0 (for UI)
- PyYAML >= 5.4.0

## 🔧 Development Status

**This is an MVP under active development.** The package is being continuously updated and generalized as part of ongoing PhD research at Northwestern University. Features and APIs may change as the research progresses.

### Roadmap
- Enhanced plugin system for more sensors
- Improved cloud-robotics integration
- Performance optimizations for real-time control

## 🤝 Contributing

This project is primarily research-focused, but contributions are welcome! Please feel free to:

- Report issues or bugs
- Suggest improvements
- Submit pull requests

## 📚 Citation

If you use this software in your research, you are welcomed cite:

```bibtex
@software{metamachine2024,
  title={Capybarish: A lightweight communication middleware designed for cloud-based robotics controllers},
  author={Chen Yu},
  year={2025},
  url={https://github.com/chenaah/capybarish}
}
```

```bibtex
@article{yu2025reconfigurable,
  title={Reconfigurable legged metamachines that run on autonomous modular legs},
  author={Yu, Chen and Matthews, David and Wang, Jingxian and Gu, Jing and Blackiston, Douglas and Rubenstein, Michael and Kriegman, Sam},
  journal={arXiv preprint arXiv:2505.00784},
  year={2025}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Chen Yu**  
PhD Student, Northwestern University  
Email: chenyu@u.northwestern.edu

---

*This middleware is part of ongoing research on evolutionary modular robots. Updates and improvements will be released as the research progresses.*