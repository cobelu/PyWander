# PyWander

## Notes
* Rather than passing around h, pass around the reference to h in the data store
* Ex: One fast worker could take another piece of work from the queue
* One queue for Sync case

## Help
* Venv (Specifically, Python3.7 for Ray):
  * `python3 -m venv venv`
  * `source venv/bin/activate`
  * `pip install -r requirements.txt`
* Default:
  * `python wander.py data/netflix.npz`
