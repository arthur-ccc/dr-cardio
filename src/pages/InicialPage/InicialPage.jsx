import { Link } from "react-router-dom"
import { FaDatabase, FaProjectDiagram, FaHeartbeat } from "react-icons/fa"
import { GiHealthNormal, GiCheckMark } from "react-icons/gi"
import "./InicialPage.css"

export default function InicialPage() {
  return (
    <div className="scroll-page">
      {/* Seção 1 - Sobre a disciplina */}
      <section className="section disciplina">
        <div className="content">
          <h1>Projeto da Disciplina de Inteligência Artificial</h1>
          <p>
            Este projeto foi desenvolvido como parte da disciplina de{" "}
            <strong>Inteligência Artificial</strong>, ministrada pelo professor{" "}
            <strong>Eanes Torres</strong> na UFCG. O objetivo é aplicar conceitos de IA
            para auxiliar no diagnóstico de doenças de forma automatizada.
          </p>
          <GiHealthNormal className="section-icon" />
        </div>
      </section>

      {/* Seção 2 - Introdução ao trabalho */}
      <section className="section introducao">
        <div className="content">
          <h1>Nosso Trabalho</h1>
          <p>
            O sistema utiliza informações clínicas coletadas em nossa base de dados
            e aplica um modelo baseado em <strong>árvore de decisão</strong> para
            analisar sintomas e sugerir possíveis diagnósticos.
          </p>
          <div className="cards">
            <div className="card">
              <FaDatabase className="card-icon" />
              <h3>Base de Dados</h3>
              <p>
                Contém centenas de registros clínicos de pacientes <span>reais</span> com sintomas variados.
              </p>
            </div>
            <div className="card">
              <FaProjectDiagram className="card-icon" />
              <h3>Árvore de Decisão</h3>
              <p>
                Analisa os dados e fornece um resultado preciso baseado em padrões.
              </p>
            </div>
            <div className="card">
              <FaHeartbeat className="card-icon" />
              <h3>Diagnóstico Automatizado</h3>
              <p>
                Oferece uma recomendação rápida para auxiliar médicos e pacientes. 
                No entanto, este diagnóstico <strong>não substitui</strong> a avaliação médica profissional.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Seção 3 - Como funciona */}
      <section className="section funcionamento">
        <div className="content">
          <h1>Como Funciona</h1>
          <div className="steps">
            <div className="step">
              <GiHealthNormal className="step-icon" />
              <h4>1. Coleta de Dados</h4>
              <p>
                O usuário informa os sintomas sentidos, seguindo os dados presentes em nossa base.
              </p>
            </div>
            <div className="step">
              <GiCheckMark className="step-icon" />
              <h4>2. Processamento</h4>
              <p>
                A árvore de decisão processa os dados usando padrões previamente treinados.
              </p>
            </div>
            <div className="step">
              <FaHeartbeat className="step-icon" />
              <h4>3. Resultado</h4>
              <p>
                O sistema apresenta um diagnóstico possível, baseado em exemplos previamente analisados.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Seção 4 - Formulário */}
      <section className="section formulario">
        <div className="content">
          <h1>Teste seu Diagnóstico!</h1>
          <p>
            Clique no botão abaixo para acessar o formulário. Preencha suas informações
            e receba uma análise baseada em nosso modelo de IA.
          </p>
          <Link to="/formulario">
            <button className="btn-formulario">Ir para o Formulário</button>
          </Link>
        </div>
      </section>
    </div>
  )
}
