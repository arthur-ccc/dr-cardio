import './App.css'

import Header from './static/header/header'
import InicialPage from './pages/InicialPage/InicialPage'
import Formulario from './pages/Formulario/Formulario'

import { Routes, Route } from 'react-router-dom'

function App() {
  return (
    <>
      <Header />

      <Routes>
        <Route path="/" element={<InicialPage />} />
        <Route path="/formulario" element={<Formulario />} />
      </Routes>
    </>
  )
}

export default App
