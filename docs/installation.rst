.. _installation:

Installation Guide
===========================


Pre-Installation
----------------------------
Before you can install the software, you must first have a working Python distribution with a package manager. For all platforms we recommend that you download and install Anaconda - a professional grade, full-blown scientific Python distribution.

To set up Anaconda, you should:

    * Download and install Anaconda (Python 3.x version, 64 bit installer is recommended) from https://www.continuum.io/downloads
    
    * Update the root Anaconda environment (type in a terminal): 
        
        ``>> conda update --all``
    
    * Activate the Anaconda root environment in a terminal as follows: 
        
        ``>> activate``


Simple Installation
----------------------------

ssms’s base code is open-sourced and freely available on `GitLab 
<https://gitlab.windenergy.dtu.dk/TOPFARM/basic-plugins/support-structure-mass-surrogate>`_ (MIT license).

* Install from PyPi.org (official releases)::
  
    pip install ssms

* Install from GitLab  (includes any recent updates)::
  
    pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/basic-plugins/support-structure-mass-surrogate.git
        


Developer Installation
-------------------------------

We highly recommend developers to install ssms into the environment created previously. The commands to clone and install ssms with developer options into the current active environment in an Anaconda Prompt are as follows::

   git clone https://gitlab.windenergy.dtu.dk/TOPFARM/basic-plugins/support-structure-mass-surrogate.git
   cd ssms
   pip install -e .
