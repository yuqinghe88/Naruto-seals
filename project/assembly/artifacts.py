"""One artifact writer; current docs/catalog are not rewritten by experiments."""
from assembly.paths import input_path
import hashlib
from assembly.io import dump
from assembly.paths import output_path


class ArtifactStore:
    def __init__(self,directory):
        self.root=output_path(directory)
        if self.root.exists() and any(self.root.iterdir()):raise ValueError('Run output must be new or empty; no silent overwrite')
        self.root.mkdir(parents=True,exist_ok=True);self.entries=[]

    def save(self,name,artifact):
        path=self.root/(name+'.json');dump(path,artifact.to_dict())
        self.entries.append({'stage':artifact.stage,'file':path.name,'sha256':hashlib.sha256(input_path(path).read_bytes()).hexdigest(),
            'rig_id':artifact.rig_id,'model_sha256':artifact.model_sha256,'authority_sha256':artifact.authority_sha256})

    def finish(self,budget,**extra):
        dump(self.root/'manifest.json',{'schema':'naruto.run_manifest.v1','artifacts':self.entries,'budget':budget.report(),**extra})
