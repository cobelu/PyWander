# PyWander

## Notes
* Rather than passing around h, pass around the reference to h in the data store
* Ex: One fast worker could take another piece of work from the queue
* One queue for Sync case

## Help
* Venv (Specifically, Python3.7 for Ray):
  * `python3.7 -m venv venv`
  * `source venv/bin/activate`
  * `pip install -r requirements.txt`
* Default:
  * `python wander.py data/netflix.npz`
  * `python wander.py -w 4 -a 0.008 -b 0.01 -l 0.05 -s data/netflix.npz`
  * `python wander.py -w 4 -a 0.0005 -b 0.05 -l 0.05 -s data/netflix.npz`
  * `[sudo] ray up support/bsn.yaml`
* Network Troubleshooting:
  * `sudo netstat -peanut | grep ":8265"`

## TODO
* Train/Test split (what are the values?)
* Bold driving fixes
* HugeWiki download/conversion
* MovieLens download/conversion
