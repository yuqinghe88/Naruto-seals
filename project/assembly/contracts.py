"""Dependency-free versioned stage contracts, independent of orchestration."""
from dataclasses import dataclass
import json

STAGES={'semantic_plan','pose_set','trajectory_controls','assembled_frames','render_result'}


class ContractError(ValueError):pass


@dataclass
class Artifact:
    stage: str
    rig_id: str
    model_sha256: str
    authority_sha256: str
    payload: dict
    provenance: dict

    def to_dict(self):
        result={'schema':'naruto.stage_artifact.v1',**self.__dict__}
        if not isinstance(self.stage,str) or self.stage not in STAGES:raise ContractError('Unknown stage')
        if not isinstance(self.payload,dict) or not isinstance(self.provenance,dict):raise ContractError('Stage payload and provenance must be objects')
        if not all(isinstance(x,str) and x for x in (self.rig_id,self.model_sha256,self.authority_sha256)):
            raise ContractError('Explicit rig, model and authority versions required')
        json.dumps(result,allow_nan=False)
        return result

    @classmethod
    def from_dict(cls,value):
        if value.get('schema')!='naruto.stage_artifact.v1':raise ContractError('Unsupported artifact version')
        result=cls(**{k:value[k] for k in cls.__dataclass_fields__});result.to_dict();return result

    def require(self,stage,rig_id=None,model_sha256=None,authority_sha256=None):
        if self.stage!=stage:raise ContractError(f'Expected {stage}, received {self.stage}')
        for name,expected in [('rig_id',rig_id),('model_sha256',model_sha256),('authority_sha256',authority_sha256)]:
            if expected is not None and getattr(self,name)!=expected:raise ContractError(name+' mismatch; explicit migration required')
        return self


class RunBudget:
    def __init__(self,fk=200,abstract=200):
        if any(isinstance(x,bool) or not isinstance(x,int) or not 1<=x<=500 for x in (fk,abstract)):
            raise ContractError('Budgets must be integers in 1..500')
        self.limits={'FK':fk,'abstract':abstract};self.used={'FK':0,'abstract':0}

    def charge(self,kind,count):
        if kind not in self.used or isinstance(count,bool) or not isinstance(count,int) or count<0:raise ContractError('Invalid budget charge')
        if self.used[kind]+count>self.limits[kind]:raise ContractError(kind+' budget exhausted')
        self.used[kind]+=count

    def report(self):return {'limits':self.limits.copy(),'used':self.used.copy(),'reset_between_stages':False}

    def require_continuation(self,snapshot):
        if not isinstance(snapshot,dict):raise ContractError('Missing inherited budget; explicit import required')
        inherited=RunBudget.resume(snapshot)
        if inherited.limits!=self.limits or any(self.used[k]<inherited.used[k] for k in self.used):
            raise ContractError('Inherited budget cannot reset or expand between stages')

    @classmethod
    def resume(cls,snapshot):
        if snapshot is None:return cls()
        result=cls(snapshot['limits']['FK'],snapshot['limits']['abstract'])
        for kind in result.used:result.charge(kind,snapshot['used'][kind])
        return result
