function TextDocumentVisualizer()
{
    var self = this;

    self._upscaleWeight = function(weight)
    {
        return Math.pow((Math.log2(1 + weight)), (1/4));
    };

    self.populateSentenceDiv = function(sentenceObj, sentenceDiv)
    {
        if (sentenceObj['isUsed'])
        {
            var sentenceWeight = self._upscaleWeight(sentenceObj['weight']);
            $(sentenceDiv).css('border-color', 'rgba(220, 53, 69, ' + sentenceWeight +')');

            var wordsArr = sentenceObj['words'];
            for (var i = 0; i < wordsArr.length; i++)
            {
                var wordObj = wordsArr[i];
                var word = wordObj['word'];
                if (word == '<GENERELATEDTOKEN>')
                {
                    word = '(GENERELATEDTOKEN-->)';
                }
                if (word == '<VARIATIONRELATEDTOKEN>')
                {
                    word = '(VARIATIONRELATEDTOKEN-->)';
                }
                var wordSpan = document.createElement('span');
                $(wordSpan).html(word);

                var wordWeight = wordObj['weight'];
                $(wordSpan).css('background-color', 'rgba(0, 123, 255, ' + sentenceWeight * self._upscaleWeight(wordWeight) +')');
                
                var spaceSpan = document.createElement('span');
                $(spaceSpan).html(' ');

                sentenceDiv.appendChild(wordSpan);
                sentenceDiv.appendChild(spaceSpan);
            }
        }
        else
        {
            $(sentenceDiv).addClass('relevantUnusedSentence');
            $(sentenceDiv).html(sentenceObj["words"]);
        }
    };

    self._renderSentence = function(geneSentenceObj, varSentenceObj, container)
    {
        var sentenceDiv = document.createElement("div");
        if (!geneSentenceObj['isRelevant'] && !varSentenceObj['isRelevant'])
        {
            $(sentenceDiv).addClass('irrelevantSentence');
            $(sentenceDiv).html(geneSentenceObj["words"]);
        }
        else
        {
            $(sentenceDiv).addClass('relevantSentence');
            if (geneSentenceObj['isRelevant'] && !varSentenceObj['isRelevant'])
            {
                self.populateSentenceDiv(geneSentenceObj, sentenceDiv);
            }
            else if (!geneSentenceObj['isRelevant'] && varSentenceObj['isRelevant'])
            {
                self.populateSentenceDiv(varSentenceObj, sentenceDiv);
            }
            else
            {
                if (geneSentenceObj['weight'] < varSentenceObj['weight'])
                {
                    self.populateSentenceDiv(varSentenceObj, sentenceDiv);
                }
                else (geneSentenceObj['weight'] > varSentenceObj['weight'])
                {
                    self.populateSentenceDiv(varSentenceObj, sentenceDiv);
                }
            }
        }
        container.appendChild(sentenceDiv);
    };

    self._round = function(x, num_decimal_places)
    {
        return Number(Math.round(x+'e'+num_decimal_places)+'e-'+num_decimal_places)
    };

    self._renderResultTable = function(trueClass, prediction, container)
    {
        var resultTable = document.createElement('table');
        $(resultTable).addClass('table table-bordered table-responsive');

        if (trueClass.length == 9)
        {
            var classNameMap = ["Likely LOF", "Likely GOF", "Neutral", "LOF", "Likely Neutral", "Inconclusive", "GOF", "Likely SOF", " SOF"];
        }
        else if (trueClass.length == 5)
        {
            classNameMap = ["LOF", "GOF", "Neutral", "Inconclusive", "SOF"];
        }
        else if (trueClass.length == 3)
        {
            classNameMap = ["Sure", "Likely", "Inconclusive"];
        }

        var thead = document.createElement('thead');
        var headtr = document.createElement('tr');
        thead.appendChild(headtr);
        var th = document.createElement('th');
        $(th).html('#');
        headtr.appendChild(th);

        var tbody = document.createElement('tbody');
        var truetr = document.createElement('tr');
        var trueth = document.createElement('th');
        $(trueth).attr('scope', 'row');
        $(trueth).html('True');
        truetr.appendChild(trueth);

        var predtr = document.createElement('tr');
        var predth = document.createElement('th');
        $(predth).attr('scope', 'row');
        $(predth).html('Prediction');
        predtr.appendChild(predth);

        tbody.appendChild(truetr);
        tbody.appendChild(predtr);

        resultTable.appendChild(thead);
        resultTable.appendChild(tbody);

        for (var i = 0; i < trueClass.length; i++)
        {
            var th = document.createElement('th');
            $(th).html(classNameMap[i]);
            headtr.appendChild(th);

            var truetd = document.createElement('td');
            $(truetd).html(trueClass[i]);
            if (trueClass[i] == 1)
            {
                $(truetd).css('background-color', 'rgb(40, 167, 69)');
            }

            var predtd = document.createElement('td');
            $(predtd).html(self._round(prediction[i], 3));
            $(predtd).css('background-color', 'rgba(40, 167, 69, ' + prediction[i] + ')');


            truetr.appendChild(truetd);
            predtr.appendChild(predtd);
        }

        container.appendChild(resultTable);
    };

    self._renderModelScores = function(log_loss, accuracy, container)
    {
        var modelScoreDiv = document.createElement("div");
        $(modelScoreDiv).addClass('modelScoresDiv border');
        var loglossDiv = document.createElement("div");
        $(loglossDiv).html('<span class="scoreLabel">Model Log Loss: </span>' + self._round(log_loss, 3));
        var accuracyDiv = document.createElement("div");
        $(accuracyDiv).html('<span class="scoreLabel">Model Accuracy: </span>' + self._round(accuracy, 3));

        modelScoreDiv.appendChild(loglossDiv);
        modelScoreDiv.appendChild(accuracyDiv);
        container.appendChild(modelScoreDiv);
    };

    self._renderScoresInfo = function(trueClass, prediction, log_loss, accuracy, container)
    {
        var scoresDiv = document.createElement("div");
        $(scoresDiv).addClass('scoresDiv');
        self._renderModelScores(log_loss, accuracy, scoresDiv);
        self._renderResultTable(trueClass, prediction, scoresDiv);
        scoresDiv.appendChild(document.createElement('hr'))

        container.appendChild(scoresDiv);
        // container.appendChild(document.createElement('hr'));
    };

    self.visualize = function(textDocument, container)
    {
        textDiv = document.createElement("div");
        $(textDiv).addClass('col');

        self._renderScoresInfo(textDocument['trueClass'], textDocument['prediction'], textDocument['logloss'], textDocument['accuracy'], textDiv);

        for (var i = 0; i < textDocument['geneText'].length; i++)
        {
            var geneSentenceObj = textDocument['geneText'][i];
            var varSentenceObj = textDocument['variationText'][i];
            self._renderSentence(geneSentenceObj, varSentenceObj, textDiv);
        }
        container.appendChild(textDiv);
    };

    self.visualizeFiltered = function(textDocument, container)
    {
        textDiv = document.createElement("div");
        $(textDiv).addClass('col');

        self._renderScoresInfo(textDocument['trueClass'], textDocument['prediction'], textDocument['logloss'], textDocument['accuracy'], textDiv);

        var usedSentenceInd = [];
        for (var i = 0; i < textDocument['geneText'].length; i++)
        {
            var geneSentenceObj = textDocument['geneText'][i];
            var varSentenceObj = textDocument['variationText'][i];
            if (geneSentenceObj['isUsed'] || varSentenceObj['isUsed'])
            {
                usedSentenceInd.push(i);
            }
        }

        usedSentenceInd.sort(function(a, b) {
            return Math.max(textDocument['geneText'][b]['weight'], textDocument['variationText'][b]['weight']) - Math.max(textDocument['geneText'][a]['weight'], textDocument['variationText'][a]['weight']);
        });

        usedSentenceInd.forEach(function(i) {
            var geneSentenceObj = textDocument['geneText'][i];
            var varSentenceObj = textDocument['variationText'][i];
            self._renderSentence(geneSentenceObj, varSentenceObj, textDiv);
        });

        container.appendChild(textDiv);
    }
}

function viewModel(stage1Results, stage2Results)
{
    var self = this;
    self.stage1Results = stage1Results;
    self.stage2Results = stage2Results;
    self.currentStageResults = self.stage1Results;
    self.modelsInd = 1;
    self.results = self.currentStageResults[self.modelsInd];
    self.tdv = new TextDocumentVisualizer();
    self.state = 'filter';

    $('#stage1label').click(function(event) {
        self.currentStageResults = self.stage1Results
        self.results = self.currentStageResults[self.modelsInd];
        self.main();
    });

    $('#stage2label').click(function(event) {
        self.currentStageResults = self.stage2Results
        self.results = self.currentStageResults[self.modelsInd];
        self.main();
    });

    $('#rawLabel').click(function(event) {
        self.modelsInd = 1;
        self.results = self.currentStageResults[self.modelsInd];
        
        var select = document.getElementById('instanceSelect');
        self.textDocuments = self.results.map(function(result) {
            return result[select.selectedIndex];
        });
        self.update();
    });

    $('#condensedLkelihoodLabel').click(function(event) {
        self.modelsInd = 0;
        self.results = self.currentStageResults[self.modelsInd];
        
        var select = document.getElementById('instanceSelect');
        self.textDocuments = self.results.map(function(result) {
            return result[select.selectedIndex];
        });
        self.update();
    });

    self.populateSelect = function(select)
    {
        $(select).empty();
        self.results[0].forEach(function(textDocument) {
            var option = document.createElement('option');
            $(option).html(textDocument['ID']);
            select.appendChild(option);
        });
    };

    self.update = function()
    {
        $('#content').empty();

        $('#instanceName').html(self.textDocuments[0]['Gene'] + '/' + self.textDocuments[0]['Variation']);

        var rowDiv = document.createElement('div');
        $(rowDiv).addClass('row');
        if (self.state == 'filter')
        {
            $('#fullView').removeClass('active');
            self.textDocuments.forEach(function(textDocument) {
                self.tdv.visualizeFiltered(textDocument, rowDiv);
            });
            $('#filterView').addClass('active');
        }
        else
        {
            $('#filterView').removeClass('active');
            self.textDocuments.forEach(function(textDocument) {
                self.tdv.visualize(textDocument, rowDiv);
            });
            $('#fullView').addClass('active');
        }

        document.getElementById('content').appendChild(rowDiv);
    }

    self.main = function()
    {
        var select = document.getElementById('instanceSelect');
        self.populateSelect(select);

        $(select).change(function(event) {
            self.textDocuments = self.results.map(function(result) {
                return result[select.selectedIndex];
            });
            self.update();
        });

        var selectedIndex = document.getElementById('instanceSelect').selectedIndex;
        self.textDocuments = self.results.map(function(result) {
            return result[select.selectedIndex];
        });
        self.update();

        $('#filterView').click(function(event) {
            self.state = 'filter';
            self.update();
        });
        $('#fullView').click(function(event) {
            self.state = 'full';
            self.update();
        });
    };
}

$(function() {
    var loadingJumbotron = document.createElement('div');
    var loadingH1 = document.createElement('h1');
    $(loadingJumbotron).addClass('jumbotron');
    $(loadingH1).addClass('display-3');
    $(loadingH1).html('Loading...');
    loadingJumbotron.appendChild(loadingH1);
    document.getElementById('content').appendChild(loadingJumbotron);

    var rawModelMetaData, rawModelGeneText, rawModelVarText;
    var condensedModelMetaData, condensedModelGeneText, condensedModelVarText;
    var likelihoodModelMetaData, likelihoodModelGeneText, likelihoodModelVarText;

    var rawModelMetaData2, rawModelGeneText2, rawModelVarText2;
    var condensedModelMetaData2, condensedModelGeneText2, condensedModelVarText2;
    var likelihoodModelMetaData2, likelihoodModelGeneText2, likelihoodModelVarText2;
    $.when(
        $.getJSON("results/rawModel_ResultsStage1_metaData.json", function(data) {
            rawModelMetaData = data;
        }),
        $.getJSON("results/rawModel_ResultsStage1_geneText.json", function(data) {
            rawModelGeneText = data;
        }),
        $.getJSON("results/rawModel_ResultsStage1_varText.json", function(data) {
            rawModelVarText = data;
        }),
        $.getJSON("results/condensedModel_ResultsStage1_metaData.json", function(data) {
            condensedModelMetaData = data;
        }),
        $.getJSON("results/condensedModel_ResultsStage1_geneText.json", function(data) {
            condensedModelGeneText = data;
        }),
        $.getJSON("results/condensedModel_ResultsStage1_varText.json", function(data) {
            condensedModelVarText = data;
        }),
        $.getJSON("results/likelihoodModel_ResultsStage1_metaData.json", function(data) {
            likelihoodModelMetaData = data;
        }),
        $.getJSON("results/likelihoodModel_ResultsStage1_geneText.json", function(data) {
            likelihoodModelGeneText = data;
        }),
        $.getJSON("results/likelihoodModel_ResultsStage1_varText.json", function(data) {
            likelihoodModelVarText = data;
        }),
        $.getJSON("results/rawModel_ResultsStage2_metaData.json", function(data) {
            rawModelMetaData2 = data;
        }),
        $.getJSON("results/rawModel_ResultsStage2_geneText.json", function(data) {
            rawModelGeneText2 = data;
        }),
        $.getJSON("results/rawModel_ResultsStage2_varText.json", function(data) {
            rawModelVarText2 = data;
        }),
        $.getJSON("results/condensedModel_ResultsStage2_metaData.json", function(data) {
            condensedModelMetaData2 = data;
        }),
        $.getJSON("results/condensedModel_ResultsStage2_geneText.json", function(data) {
            condensedModelGeneText2 = data;
        }),
        $.getJSON("results/condensedModel_ResultsStage2_varText.json", function(data) {
            condensedModelVarText2 = data;
        }),
        $.getJSON("results/likelihoodModel_ResultsStage2_metaData.json", function(data) {
            likelihoodModelMetaData2 = data;
        }),
        $.getJSON("results/likelihoodModel_ResultsStage2_geneText.json", function(data) {
            likelihoodModelGeneText2 = data;
        }),
        $.getJSON("results/likelihoodModel_ResultsStage2_varText.json", function(data) {
            likelihoodModelVarText2 = data;
        })
    ).then(function() {
        if (rawModelMetaData && rawModelGeneText && rawModelVarText)
        {
            for (var i = 0; i < rawModelMetaData.length; i++)
            {
                rawModelMetaData[i]['geneText'] = rawModelGeneText[i];
                rawModelMetaData[i]['variationText'] = rawModelVarText[i];
            }
        }
        if (condensedModelMetaData, condensedModelGeneText, condensedModelVarText)
        {
            for (var i = 0; i < condensedModelMetaData.length; i++)
            {
                condensedModelMetaData[i]['geneText'] = condensedModelGeneText[i];
                condensedModelMetaData[i]['variationText'] = condensedModelVarText[i];
            }
        }
        if (likelihoodModelMetaData, likelihoodModelGeneText, likelihoodModelVarText)
        {
            for (var i = 0; i < likelihoodModelMetaData.length; i++)
            {
                likelihoodModelMetaData[i]['geneText'] = likelihoodModelGeneText[i];
                likelihoodModelMetaData[i]['variationText'] = likelihoodModelVarText[i];
            }
        }
        if (rawModelMetaData2 && rawModelGeneText2 && rawModelVarText2)
        {
            for (var i = 0; i < rawModelMetaData2.length; i++)
            {
                rawModelMetaData2[i]['geneText'] = rawModelGeneText2[i];
                rawModelMetaData2[i]['variationText'] = rawModelVarText2[i];
            }
        }
        if (condensedModelMetaData2, condensedModelGeneText2, condensedModelVarText2)
        {
            for (var i = 0; i < condensedModelMetaData2.length; i++)
            {
                condensedModelMetaData2[i]['geneText'] = condensedModelGeneText2[i];
                condensedModelMetaData2[i]['variationText'] = condensedModelVarText2[i];
            }
        }
        if (likelihoodModelMetaData2, likelihoodModelGeneText2, likelihoodModelVarText2)
        {
            for (var i = 0; i < likelihoodModelMetaData2.length; i++)
            {
                likelihoodModelMetaData2[i]['geneText'] = likelihoodModelGeneText2[i];
                likelihoodModelMetaData2[i]['variationText'] = likelihoodModelVarText2[i];
            }
        }

        $('#content').empty();
        vm = new viewModel([[condensedModelMetaData, likelihoodModelMetaData], [rawModelMetaData]], [[condensedModelMetaData2, likelihoodModelMetaData2], [rawModelMetaData2]]);
        vm.main();
    });
})