from pydantic import BaseModel, Field

class StudentInput(BaseModel):
    idade: int = Field(..., description="Idade do aluno em anos")
    sexo: str = Field(..., description="Sexo: 'M' ou 'F'")
    tipo_escola_medio: str = Field(..., description="Tipo de escola do ensino médio: 'publica' ou 'privada'")
    nota_enem: float = Field(..., description="Nota ENEM")
    renda_familiar: float = Field(..., description="Renda familiar (em salários mínimos ou mesma unidade utilizada)")
    trabalha: int = Field(..., description="0 = não trabalha, 1 = trabalha")
    horas_trabalho_semana: float = Field(..., description="Horas trabalhadas por semana")
    cra_1_sem: float = Field(..., description="CRA/IRA do 1º semestre")
    reprovacoes_1_sem: int = Field(..., description="Número de reprovações no 1º semestre")
    bolsista: int = Field(..., description="0 = não bolsista, 1 = bolsista")
    distancia_campus_km: float = Field(..., description="Distância até o campus em km")
