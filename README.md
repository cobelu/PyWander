# PyWander

## Notes
* Rather than passing around h, pass around the reference to h in the data store
* Ex: One fast worker could take another piece of work from the queue
* One queue for Sync case
