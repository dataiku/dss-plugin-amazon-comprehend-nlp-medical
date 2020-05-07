PLUGIN_VERSION=1.0.0
PLUGIN_ID=amazon-comprehend-nlp

plugin:
	cat plugin.json|json_pp > /dev/null
	rm -rf dist
	mkdir dist
	zip --exclude "*.pyc" -r dist/dss-plugin-${PLUGIN_ID}-${PLUGIN_VERSION}.zip plugin.json python-lib custom-recipes parameter-sets code-env
