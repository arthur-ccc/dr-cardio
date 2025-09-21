import React, { useState } from "react";
import "./Formulario.css";

export default function Formulario() {
  const [step, setStep] = useState(0);

  const [formData, setFormData] = useState({
    form_age: "",
    form_familyHistory: false,
    // Clínicos
    form_erythema: 0,
    form_scaling: 0,
    form_definiteBorders: 0,
    form_itching: 0,
    form_koebnerPhenomenon: 0,
    form_polygonalPapules: 0,
    form_follicularPapules: 0,
    form_oralMucosalInvolvement: 0,
    form_kneeElbowInvolvement: 0,
    form_scalpInvolvement: 0,
    // Histopatológicos
    form_melaninIncontinence: 0,
    form_eosinophilsInfiltrate: 0,
    form_pnlInfiltrate: 0,
    form_fibrosisPapillaryDermis: 0,
    form_exocytosis: 0,
    form_acanthosis: 0,
    form_hyperkeratosis: 0,
    form_parakeratosis: 0,
    form_clubbingReteRidges: 0,
    form_elongationReteRidges: 0,
    form_thinningSuprapapillaryEpidermis: 0,
    form_spongiformPustule: 0,
    form_munroMicroabcess: 0,
    form_focalHypergranulosis: 0,
    form_disappearanceGranularLayer: 0,
    form_vacuolisationBasalLayer: 0,
    form_spongiosis: 0,
    form_sawToothRete: 0,
    form_follicularHornPlug: 0,
    form_perifollicularParakeratosis: 0,
    form_inflammatoryMononuclearInfiltrate: 0,
    form_bandLikeInfiltrate: 0
  });

  const atributosTraduzidos = {
    form_erythema: "Eritema",
    form_scaling: "Descamação",
    form_definiteBorders: "Bordas definidas",
    form_itching: "Coceira",
    form_koebnerPhenomenon: "Fenômeno de Koebner",
    form_polygonalPapules: "Pápulas poligonais",
    form_follicularPapules: "Pápulas foliculares",
    form_oralMucosalInvolvement: "Envolvimento da mucosa oral",
    form_kneeElbowInvolvement: "Envolvimento de joelhos e cotovelos",
    form_scalpInvolvement: "Envolvimento do couro cabeludo",
    form_familyHistory: "Histórico familiar",
    form_age: "Idade",
    form_melaninIncontinence: "Incontinência de melanina",
    form_eosinophilsInfiltrate: "Eosinófilos no infiltrado",
    form_pnlInfiltrate: "PNL infiltrado",
    form_fibrosisPapillaryDermis: "Fibrose da derme papilar",
    form_exocytosis: "Exocitose",
    form_acanthosis: "Acantose",
    form_hyperkeratosis: "Hiperqueratose",
    form_parakeratosis: "Paraceratose",
    form_clubbingReteRidges: "Alargamento das cristas epiteliais",
    form_elongationReteRidges: "Alongamento das cristas epiteliais",
    form_thinningSuprapapillaryEpidermis: "Afinamento da epiderme suprapapilar",
    form_spongiformPustule: "Pústula espongiforme",
    form_munroMicroabcess: "Microabscesso de Munro",
    form_focalHypergranulosis: "Hipergranulose focal",
    form_disappearanceGranularLayer: "Desaparecimento da camada granulosa",
    form_vacuolisationBasalLayer: "Vacuolização da camada basal",
    form_spongiosis: "Espongiose",
    form_sawToothRete: "Aspecto em serra das cristas epiteliais",
    form_follicularHornPlug: "Plug folicular",
    form_perifollicularParakeratosis: "Paraceratose perifolicular",
    form_inflammatoryMononuclearInfiltrate: "Infiltrado mononuclear inflamatório",
    form_bandLikeInfiltrate: "Infiltrado em faixa",
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === "checkbox" ? checked : value
    });
  };

  const nextStep = () => setStep((prev) => prev + 1);
  const prevStep = () => setStep((prev) => prev - 1);

  const renderSelect = (name) => (
    <div className="form-group" key={name}>
      <label>{atributosTraduzidos[name]}</label>
      <select name={name} value={formData[name]} onChange={handleChange}>
        <option value={0}>0 - Não presente</option>
        <option value={1}>1 - Leve</option>
        <option value={2}>2 - Moderado</option>
        <option value={3}>3 - Intenso</option>
      </select>
    </div>
  );

  const todosAtributos = Object.keys(formData).filter(
    (attr) => attr !== "form_age" && attr !== "form_familyHistory"
  );

  return (
    <div className="form-container">
      {step === 0 && (
        <div className="step-intro">
          <h2>Bem-vindo ao formulário de diagnóstico</h2>
          <p>
            Agora você vai preencher este formulário para tentarmos diagnosticar sua doença.
            Responda cada pergunta com atenção. Cada sintoma deve ser avaliado usando a escala de 0 a 3:
          </p>
          <ul>
            <li><strong>0:</strong> Não presente</li>
            <li><strong>1:</strong> Leve</li>
            <li><strong>2:</strong> Moderado</li>
            <li><strong>3:</strong> Intenso</li>
          </ul>
          <button className="btn-submit" onClick={nextStep}>Iniciar formulário</button>
        </div>
      )}

      {step === 1 && (
        <div className="step-age">
          <h2>Vamos começar!</h2>
          <p>Digite sua idade:</p>
          <input
            type="number"
            name="form_age"
            value={formData.form_age}
            onChange={handleChange}
            placeholder="Digite sua idade"
            min="0"
          />

          <div className="family-history">
            <p>Sua família já apresentou histórico de doenças eritemato-esquamosas? Marque aqui se sim:</p>
            <input
              type="checkbox"
              name="form_familyHistory"
              checked={formData.form_familyHistory}
              onChange={handleChange}
            />
          </div>

          <div className="buttons">
            <button className="btn-submit" onClick={prevStep}>Voltar</button>
            <button
              className="btn-submit"
              onClick={nextStep}
              disabled={!formData.form_age}
            >
              Próximo
            </button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="step-form">
          <h2>Preencha os sintomas</h2>
          {todosAtributos.map(renderSelect)}
          <div className="buttons">
            <button className="btn-submit" onClick={prevStep}>Voltar</button>
            <button className="btn-submit" onClick={nextStep}>Próximo</button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="step-result">
          <h2>Diagnóstico previsto (simulado)</h2>
          <pre>{JSON.stringify(formData, null, 2)}</pre>
          <button className="btn-submit" onClick={prevStep}>Voltar</button>
        </div>
      )}
    </div>
  );
}
