from dataclasses import dataclass

@dataclass
class Student:
    idade: int
    sexo: str
    tipo_escola_medio: str
    nota_enem: float
    renda_familiar: float
    trabalha: int
    horas_trabalho_semana: float
    cra_1_sem: float
    reprovacoes_1_sem: int
    bolsista: int
    distancia_campus_km: float
