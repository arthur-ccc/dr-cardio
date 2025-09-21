import { useState } from "react"
import "./Header.css"

export default function Header() {
  const [open, setOpen] = useState(false)

  return (
    <>
      <header className="header-container">
        {/* Logo */}
        <div className="header-logo">
          <img src="./UFCG-lateral-SemFundo.png" alt="UFCG logo" />
        </div>

        {/* Texto */}
        <div className="header-text">
          <h1>Dr. Derma</h1>
          <h2>
            Seu <span>médico</span> virtual!
          </h2>
        </div>

        {/* Botão para abrir modal */}
        <nav className="header-nav">
          <button className="btn-integrantes" onClick={() => setOpen(true)}>
            Integrantes
          </button>
        </nav>
      </header>

      {/* Modal */}
      {open && (
        <div className="modal-overlay" onClick={() => setOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Integrantes do Grupo</h3>
            <ul>
              <li>Guilherme Jose</li>
              <li>Maria Luiza</li>
              <li>Rafael Alencar</li>
              <li>Isaque Esdras</li>
              <li>Jeferson Abrantes</li>
              <li>Arthur Fernandes</li>
            </ul>
            <button className="btn-fechar" onClick={() => setOpen(false)}>
              Fechar
            </button>
          </div>
        </div>
      )}
    </>
  )
}
