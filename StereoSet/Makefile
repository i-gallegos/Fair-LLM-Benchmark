.PHONY: sync download-results

sync:
	rsync --info=progress2 -urltv --delete \
		--filter=":- .gitignore" \
		-e ssh . mila:~/workspace/debias-eval

download-results:
	rsync --info=progress2 -urltv --delete \
		-e ssh mila:~/scratch/debias-eval/results/ ./results
